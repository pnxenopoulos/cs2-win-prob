"""Fit and evaluate Kumo from a versioned round-state dataset."""

from __future__ import annotations

import json
from argparse import ArgumentParser
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from cs_win_prob.data_preparation import (
    BASE_FEATURE_COLUMNS,
    ROUND_END_REASONS,
    available_feature_columns,
    sha256_file,
)
from cs_win_prob.kumo import (
    DEFAULT_CONTEXT_ROWS,
    DEFAULT_ESTIMATORS,
    DEFAULT_SEED,
    TARGET,
    WINNER_TARGET,
    fit_kumo,
    predict_probabilities,
    sample_context_rows,
    save_kumo_context,
)

if TYPE_CHECKING:
    from sdm.models import KumoTabular

_WINNER_BY_REASON = {
    "bomb_defusal": "CT",
    "ct_eliminated": "T",
    "t_eliminated": "CT",
    "bomb_exploded": "T",
    "time_ran_out": "CT",
}


def _split_counts(dataset: pl.LazyFrame) -> dict[str, dict[str, int]]:
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


def _evaluate(
    model: KumoTabular,
    test: pl.LazyFrame,
    feature_columns: tuple[str, ...],
    target: str,
) -> dict[str, float]:
    classes = (
        ("CT", "T") if target == WINNER_TARGET else tuple(sorted(ROUND_END_REASONS))
    )
    class_indices = {name: index for index, name in enumerate(classes)}
    rows = correct = 0
    log_loss = brier = 0.0
    for batch in test.select(*feature_columns, target).collect_batches(chunk_size=256):
        actual = batch[target]
        if set(actual.unique()) - set(classes):
            raise ValueError(f"test split contains an invalid {target}")
        probabilities = predict_probabilities(model, batch)
        if probabilities.height != batch.height or set(probabilities.columns) - set(
            classes
        ):
            raise ValueError("Kumo returned an unexpected probability table")
        # A class absent from the training context has zero predicted probability.
        scored = probabilities.select(
            pl.col(name).cast(pl.Float64)
            if name in probabilities.columns
            else pl.lit(0.0).alias(name)
            for name in classes
        ).with_columns(actual.alias("_actual"))
        actual_label = pl.col("_actual")
        actual_index = actual_label.replace_strict(
            class_indices, return_dtype=pl.UInt32
        )
        predicted_index = pl.concat_list(classes).list.arg_max()
        true_probability = pl.sum_horizontal(
            pl.when(actual_label == name).then(pl.col(name)).otherwise(0.0)
            for name in classes
        )
        brier_classes = ("CT",) if target == WINNER_TARGET else classes
        squared_error = pl.sum_horizontal(
            (pl.col(name) - (actual_label == name).cast(pl.Float64)) ** 2
            for name in brier_classes
        )
        batch_correct, batch_log_loss, batch_brier = scored.select(
            correct=(predicted_index == actual_index).sum(),
            log_loss=-true_probability.clip(1e-15, 1.0).log().sum(),
            brier=squared_error.sum(),
        ).row(0)
        correct += batch_correct
        log_loss += batch_log_loss
        brier += batch_brier
        rows += batch.height
    return {
        "accuracy": correct / rows,
        "log_loss": log_loss / rows,
        "brier_score": brier / rows,
    }


def fit_release(
    dataset_path: Path,
    output_dir: Path,
    *,
    context_rows: int = DEFAULT_CONTEXT_ROWS,
    num_estimators: int = DEFAULT_ESTIMATORS,
    seed: int = DEFAULT_SEED,
    device: str | None = None,
    target: str = TARGET,
    baseline_features: bool = False,
) -> dict[str, object]:
    """Save a fitted context and metrics evaluated only on held-out demos."""
    if target not in {TARGET, WINNER_TARGET}:
        raise ValueError(f"unsupported Kumo target: {target}")
    context_name = (
        "round-end-reason-context.parquet"
        if target == TARGET
        else "round-win-context.parquet"
    )
    context_path = output_dir / context_name
    metrics_path = output_dir / "metrics.json"
    if dataset_path.resolve() in {context_path.resolve(), metrics_path.resolve()}:
        raise ValueError("the dataset and output files must be different")
    dataset = pl.scan_parquet(dataset_path, glob=False, hive_partitioning=False)
    if baseline_features:
        dataset = dataset.select(
            "demo_id",
            "demo_filename",
            "demo_sha256",
            "split",
            *BASE_FEATURE_COLUMNS,
            TARGET,
        )
    if target == WINNER_TARGET:
        dataset = dataset.with_columns(
            pl.col(TARGET)
            .replace_strict(_WINNER_BY_REASON, return_dtype=pl.String)
            .alias(WINNER_TARGET)
        )
    feature_columns = available_feature_columns(dataset.collect_schema().names())
    counts = _split_counts(dataset)
    context = sample_context_rows(
        dataset.filter(pl.col("split") == "train"),
        max_rows=context_rows,
        seed=seed,
        target=target,
    )
    print(
        f"Fitting Kumo with {context.height} context rows from {counts['train']['demos']} training demos",
        flush=True,
    )
    model = fit_kumo(
        context,
        num_estimators=num_estimators,
        seed=seed,
        device=device,
        target=target,
    )
    print(
        f"Evaluating {counts['test']['rows']} rows from {counts['test']['demos']} held-out demos",
        flush=True,
    )
    evaluation = _evaluate(
        model, dataset.filter(pl.col("split") == "test"), feature_columns, target
    )
    report: dict[str, object] = {
        "dataset_sha256": sha256_file(dataset_path),
        "model": "KumoTabular small",
        "structured_data_models_version": version("structured-data-models"),
        "target": target,
        "features": list(feature_columns),
        "device": str(next(model.parameters()).device),
        "num_estimators": num_estimators,
        "seed": seed,
        "context_rows": context.height,
        "train": counts["train"],
        "test": {**counts["test"], **evaluation},
    }
    metrics = json.dumps(report, indent=2, allow_nan=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_kumo_context(
        context,
        context_path,
        num_estimators=num_estimators,
        seed=seed,
        target=target,
    )
    metrics_path.write_text(metrics + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = ArgumentParser(
        description="Fit and evaluate Kumo from a combined release Parquet."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-rows", type=int, default=DEFAULT_CONTEXT_ROWS)
    parser.add_argument("--num-estimators", type=int, default=DEFAULT_ESTIMATORS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--target", choices=(TARGET, WINNER_TARGET), default=TARGET)
    parser.add_argument(
        "--baseline-features",
        action="store_true",
        help="Use the original game-state features for comparison",
    )
    parser.add_argument(
        "--device", help="PyTorch device (default: CUDA if available, otherwise CPU)"
    )
    args = parser.parse_args()
    try:
        report = fit_release(
            args.dataset,
            args.output_dir,
            context_rows=args.context_rows,
            num_estimators=args.num_estimators,
            seed=args.seed,
            device=args.device,
            target=args.target,
            baseline_features=args.baseline_features,
        )
    except (ValueError, OSError, pl.exceptions.PolarsError) as error:
        parser.error(str(error))
    print(json.dumps(report["test"], indent=2))
