"""Combine per-demo Parquet files into a release dataset."""

import json
import re
from argparse import ArgumentParser
from math import ceil
from pathlib import Path
from random import Random
from tempfile import NamedTemporaryFile

import polars as pl

from cs_win_prob.data_preparation import (
    DEMO_METADATA_KEY,
    available_feature_columns,
    sha256_file,
)
from cs_win_prob.kumo import DEFAULT_SEED, TARGET

DEFAULT_TEST_FRACTION = 0.2
DATASET_FILENAME = "cs2-round-states.parquet"


def split_counts(dataset: pl.LazyFrame) -> dict[str, dict[str, int]]:
    """Validate demo provenance and count the two dataset splits."""
    demos = (
        dataset.group_by("demo_id", "demo_filename", "demo_sha256", "split")
        .len()
        .collect(engine="streaming")
    )
    invalid = demos.filter(
        ~pl.col("demo_sha256").str.contains(r"^[0-9a-f]{64}$").fill_null(False)
        | ~(pl.col("demo_id") == pl.col("demo_sha256")).fill_null(False)
        | (pl.col("demo_filename").fill_null("") == "")
    )
    if not invalid.is_empty():
        raise ValueError(
            "dataset must include a filename and SHA-256 demo ID for every row"
        )
    if demos["demo_id"].n_unique() != demos.height:
        raise ValueError(
            "each demo must have consistent provenance and belong to exactly one split"
        )
    if set(demos["split"]) != {"train", "test"}:
        raise ValueError("dataset must contain nonempty train and test splits")
    counts = demos.group_by("split").agg(demos=pl.len(), rows=pl.col("len").sum())
    return {
        split: {"demos": demo_count, "rows": rows}
        for split, demo_count, rows in counts.iter_rows()
    }


def _demo_provenance(path: Path) -> tuple[str, str]:
    encoded = pl.read_parquet_metadata(path).get(DEMO_METADATA_KEY)
    if encoded is None:
        raise ValueError(
            f"{path} has no demo provenance; regenerate it with cs-win-prob or cs-win-prob-train-demos"
        )
    metadata = json.loads(encoded)
    if isinstance(metadata, dict):
        filename, sha256 = metadata.get("filename"), metadata.get("sha256")
        if (
            isinstance(filename, str)
            and filename
            and isinstance(sha256, str)
            and re.fullmatch(r"[0-9a-f]{64}", sha256)
        ):
            return filename, sha256
    raise ValueError(f"invalid demo provenance in {path}")


def _reference_split(path: Path) -> dict[str, str]:
    assignments = (
        pl.scan_parquet(path, glob=False).select("demo_id", "split").unique().collect()
    )
    if (
        assignments["demo_id"].null_count()
        or assignments["demo_id"].n_unique() != assignments.height
        or set(assignments["split"]) != {"train", "test"}
    ):
        raise ValueError(
            "reference dataset must assign every demo to exactly one train or test split"
        )
    return dict(assignments.iter_rows())


def export_dataset(
    states_dir: Path,
    output: Path,
    *,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    seed: int = DEFAULT_SEED,
    split_from: Path | None = None,
) -> tuple[int, int]:
    """Write all prepared rows with stable demo IDs and return train/test demo counts."""
    if not states_dir.is_dir():
        raise ValueError(
            f"expected a directory of per-demo Parquet files: {states_dir}"
        )
    if not 0 < test_fraction < 1:
        raise ValueError("test fraction must be between 0 and 1")
    if output.suffix.lower() != ".parquet":
        raise ValueError("dataset output must be a .parquet file")

    output_path = output.resolve()
    files = {
        path.resolve()
        for path in states_dir.rglob("*.parquet")
        if path.is_file() and path.resolve() != output_path
    }
    if not files:
        raise ValueError(f"no per-demo Parquet files found in {states_dir}")

    provenance = []
    reference = _reference_split(split_from) if split_from is not None else None
    split_metadata = (
        {"reference_dataset_sha256": sha256_file(split_from)}
        if split_from is not None
        else {"seed": seed, "test_fraction": test_fraction}
    )
    for path in sorted(files):
        filename, sha256 = _demo_provenance(path)
        if reference is None or sha256 in reference:
            provenance.append((sha256, filename, str(path)))
    if not provenance:
        raise ValueError("no prepared demos match the reference dataset")
    provenance.sort()
    source = pl.scan_parquet(
        [path for _, _, path in provenance],
        glob=False,
        hive_partitioning=False,
        include_file_paths="_source",
    )
    populated = set(
        source.select("_source").unique().collect(engine="streaming")["_source"]
    )
    demos = [entry for entry in provenance if entry[2] in populated]
    if len(demos) < 2:
        raise ValueError(
            "at least two demos with states are needed for a train/test split"
        )
    demo_ids = [sha256 for sha256, _, _ in demos]
    if len(set(demo_ids)) != len(demo_ids):
        raise ValueError(
            "found duplicate demo hashes; provide one prepared Parquet per demo"
        )
    if reference is None:
        test_count = min(len(demos) - 1, ceil(test_fraction * len(demos)))
        test_ids = set(Random(seed).sample(demo_ids, test_count))
        splits = ["test" if sha256 in test_ids else "train" for sha256 in demo_ids]
    else:
        missing = set(reference) - set(demo_ids)
        if missing:
            raise ValueError(
                f"missing prepared rows for {len(missing)} demos in the reference dataset"
            )
        splits = [reference[sha256] for sha256 in demo_ids]
    test_count = splits.count("test")
    assignments = pl.DataFrame(
        {
            "_source": [path for _, _, path in demos],
            "demo_id": demo_ids,
            "demo_filename": [filename for _, filename, _ in demos],
            "demo_sha256": demo_ids,
            "split": splits,
        }
    )
    source_columns = source.collect_schema().names()
    dataset = source.join(
        assignments.lazy(), on="_source", how="inner", maintain_order="left"
    ).select(
        "demo_id",
        "demo_filename",
        "demo_sha256",
        *(("tick",) if "tick" in source_columns else ()),
        "split",
        *available_feature_columns(source_columns),
        TARGET,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent, delete=False) as file:
        staged = Path(file.name)
    try:
        dataset.sink_parquet(
            staged,
            metadata={"cs_win_prob.dataset.v1": json.dumps(split_metadata)},
        )
        staged.replace(output)
    finally:
        staged.unlink(missing_ok=True)
    return len(demos) - test_count, test_count


def main() -> None:
    parser = ArgumentParser(
        description="Export a combined demo-level train/test Parquet dataset."
    )
    parser.add_argument(
        "states_dir", type=Path, help="Directory with one prepared Parquet per demo"
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help=f"Release dataset path, typically {DATASET_FILENAME}",
    )
    parser.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--split-from",
        type=Path,
        help="Keep exactly the demos and split assignments from this dataset",
    )
    args = parser.parse_args()
    try:
        train, test = export_dataset(
            args.states_dir,
            args.output,
            test_fraction=args.test_fraction,
            seed=args.seed,
            split_from=args.split_from,
        )
    except (ValueError, OSError, pl.exceptions.PolarsError) as error:
        parser.error(str(error))
    print(f"Saved {args.output} with {train} training demos and {test} held-out demos")
