"""Describe and verify versioned dataset and model release assets."""

import json
import re
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path

import polars as pl

from cs_win_prob.data_preparation import (
    BASE_FEATURE_COLUMNS,
    ROUND_END_REASONS,
    sha256_file,
)
from cs_win_prob.dataset import DATASET_FILENAME, split_counts
from cs_win_prob.kumo import TARGET

DATASET_MANIFEST = "dataset-manifest.json"
MODEL_MANIFEST = "model-manifest.json"
MODEL_CONTEXT = "round-end-reason-context.parquet"
_SNAPSHOT_KEY = ("demo_sha256", "tick")
_VERSION = r"[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z]+(?:[.-][0-9A-Za-z]+)*)?"


def _version(tag: str, prefix: str) -> str:
    match = re.fullmatch(re.escape(prefix) + rf"/v({_VERSION})", tag)
    if match is None:
        raise ValueError(f"expected a release tag of the form {prefix}/v1.0.0")
    return match[1]


def dataset_manifest(dataset_path: Path, tag: str) -> dict[str, object]:
    """Describe the tabular release and hash its exact held-out snapshot keys."""
    version = _version(tag, "data/round-states-tabular")
    if dataset_path.name != DATASET_FILENAME:
        raise ValueError(f"the tabular release asset must be named {DATASET_FILENAME}")
    dataset = pl.scan_parquet(dataset_path, glob=False, hive_partitioning=False)
    schema = dataset.collect_schema()
    if "tick" not in schema or not schema["tick"].is_integer():
        raise ValueError(
            "the release dataset needs original demo ticks; regenerate the prepared states"
        )
    missing = {*BASE_FEATURE_COLUMNS, TARGET} - set(schema.names())
    if missing:
        raise ValueError(f"dataset is missing columns: {', '.join(sorted(missing))}")
    splits = split_counts(dataset)
    rows, unique_keys, missing_ticks = (
        dataset.select(
            pl.len(),
            pl.struct(_SNAPSHOT_KEY).n_unique(),
            pl.col("tick").is_null().any(),
        )
        .collect()
        .row(0)
    )
    if missing_ticks or rows != unique_keys:
        raise ValueError("each release row must have a unique (demo_sha256, tick) key")
    reasons = set(dataset.select(TARGET).unique().collect()[TARGET])
    if reasons - ROUND_END_REASONS:
        raise ValueError(f"dataset contains an invalid {TARGET}")

    heldout = (
        dataset.filter(pl.col("split") == "test")
        .select(_SNAPSHOT_KEY)
        .sort(_SNAPSHOT_KEY)
        .collect()
    )
    snapshot_hash = sha256()
    for demo_id, tick in heldout.iter_rows():
        snapshot_hash.update(f"{demo_id}:{tick}\n".encode("ascii"))
    return {
        "schema_version": 1,
        "name": "round-states-tabular",
        "version": version,
        "tag": tag,
        "asset": DATASET_FILENAME,
        "sha256": sha256_file(dataset_path),
        "target": TARGET,
        "snapshot_key": list(_SNAPSHOT_KEY),
        "split": {
            **splits,
            "test_demo_sha256": sorted(heldout["demo_sha256"].unique()),
            "test_snapshot_keys_sha256": snapshot_hash.hexdigest(),
        },
    }


def model_manifest(
    dataset_path: Path,
    dataset_tag: str,
    model_tag: str,
    training_commit: str,
    target: str,
) -> dict[str, object]:
    """Validate the released dataset before fitting a named Kumo model version."""
    version = _version(model_tag, "model/kumo-round-end")
    if target != TARGET:
        raise ValueError(f"model/kumo-round-end requires the {TARGET} target")
    if not re.fullmatch(r"[0-9a-f]{40}", training_commit):
        raise ValueError("training commit must be a full Git commit SHA")
    dataset = dataset_manifest(dataset_path, dataset_tag)
    released = json.loads(
        dataset_path.with_name(DATASET_MANIFEST).read_text(encoding="utf-8")
    )
    if dataset != released:
        raise ValueError(
            "dataset manifest does not match the requested tag and Parquet contents"
        )
    return {
        "schema_version": 1,
        "name": "kumo-round-end",
        "version": version,
        "tag": model_tag,
        "target": target,
        "training_commit": training_commit,
        "dataset": dataset,
    }


def write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = ArgumentParser(
        description="Write the manifest for a versioned tabular dataset release."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--tag",
        required=True,
        help="Dataset release tag, such as data/round-states-tabular/v1.0.0",
    )
    args = parser.parse_args()
    try:
        path = args.dataset.with_name(DATASET_MANIFEST)
        write_manifest(path, dataset_manifest(args.dataset, args.tag))
    except (ValueError, OSError, pl.exceptions.PolarsError) as error:
        parser.error(str(error))
    print(f"Saved {path}")
