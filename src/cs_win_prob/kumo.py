"""Fit a KumoTabular classifier for CS2 round-end reasons."""

from __future__ import annotations

import json
from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Literal, cast

import polars as pl

from cs_win_prob.data_preparation import ROUND_END_REASONS, available_feature_columns

if TYPE_CHECKING:
    import torch
    from sdm import TableTensor
    from sdm.models import KumoTabular

DEFAULT_ESTIMATORS = 8
DEFAULT_CONTEXT_ROWS = 2048
DEFAULT_SEED = 42
TARGET = "round_end_reason"
WINNER_TARGET = "round_winner"
_TARGET_LABELS = {TARGET: ROUND_END_REASONS, WINNER_TARGET: frozenset({"CT", "T"})}
_CATEGORICAL_COLUMNS = {"map", "phase", "bombsite", *_TARGET_LABELS}
_CONTEXT_METADATA = "cs_win_prob.kumo.v1"


def _parquet_paths(inputs: Sequence[Path], output: Path) -> list[Path]:
    paths = set()
    output = output.resolve()
    for path in inputs:
        if path.is_dir():
            paths.update(
                file.resolve() for file in path.rglob("*.parquet") if file.is_file()
            )
        elif path.is_file() and path.suffix.lower() == ".parquet":
            if path.resolve() == output:
                raise ValueError(
                    "the output must be a different file from the training inputs"
                )
            paths.add(path.resolve())
        else:
            raise ValueError(f"expected a Parquet file or directory: {path}")
    paths.discard(output)
    if not paths:
        raise ValueError("no training Parquet files found")
    return sorted(paths)


def _validate_labels(frame: pl.DataFrame, target: str) -> None:
    if set(frame[target].unique()) - _TARGET_LABELS[target]:
        raise ValueError(f"{target} contains a null or unsupported label")


def sample_context(
    paths: Sequence[Path],
    *,
    max_rows: int = DEFAULT_CONTEXT_ROWS,
    seed: int = DEFAULT_SEED,
    target: str = TARGET,
) -> pl.DataFrame:
    """Sample a bounded context from Parquet files, retaining each observed class."""
    if not paths:
        raise ValueError("no training Parquet files found")
    return sample_context_rows(
        pl.scan_parquet(list(paths), glob=False, hive_partitioning=False),
        max_rows=max_rows,
        seed=seed,
        target=target,
    )


def sample_context_rows(
    source: pl.LazyFrame,
    *,
    max_rows: int = DEFAULT_CONTEXT_ROWS,
    seed: int = DEFAULT_SEED,
    target: str = TARGET,
) -> pl.DataFrame:
    """Sample from a lazy frame of eligible training rows."""
    if max_rows < 2:
        raise ValueError("context size must be at least 2")
    if target not in _TARGET_LABELS:
        raise ValueError(f"unsupported Kumo target: {target}")

    columns = (*available_feature_columns(source.collect_schema().names()), target)
    context = None
    rows_seen = 0
    hash_seed = seed % (1 << 64)
    for batch in source.select(columns).collect_batches(chunk_size=16384):
        _validate_labels(batch, target)
        batch = batch.with_columns(
            _row=pl.int_range(rows_seen, rows_seen + batch.height, dtype=pl.UInt64)
        ).with_columns(_priority=pl.col("_row").hash(seed=hash_seed))
        rows_seen += batch.height
        candidates = batch if context is None else pl.concat([context, batch])
        candidates = candidates.sort("_priority", "_row")
        if candidates.height <= max_rows:
            context = candidates
            continue

        # Reserve the highest-priority row per class, then fill the remaining slots.
        representatives = candidates.unique(target, keep="first", maintain_order=True)
        if representatives.height > max_rows:
            raise ValueError(
                "context size must be at least the number of observed classes"
            )
        remainder = candidates.join(
            representatives.select("_row"), on="_row", how="anti", maintain_order="left"
        )
        context = pl.concat(
            [representatives, remainder.head(max_rows - representatives.height)]
        )

    if context is None or context[target].n_unique() < 2:
        raise ValueError(
            f"training data must contain at least two classes for {target}"
        )
    return context.sort("_priority", "_row").select(columns)


def _table(frame: pl.DataFrame, device: torch.device) -> TableTensor:
    from sdm import TableTensor

    stypes: dict[str, Literal["categorical", "numerical"]] = {
        name: "categorical" if name in _CATEGORICAL_COLUMNS else "numerical"
        for name in frame.columns
    }
    return TableTensor.from_arrow(frame.to_arrow(), stypes=stypes, device=device)


def fit_kumo(
    context: pl.DataFrame,
    *,
    num_estimators: int = DEFAULT_ESTIMATORS,
    seed: int = DEFAULT_SEED,
    device: str | None = None,
    target: str = TARGET,
) -> KumoTabular:
    """Fit pretrained KumoTabular small to labeled game states in context."""
    import torch
    from sdm.models import KumoTabular

    if num_estimators < 1:
        raise ValueError("number of estimators must be at least 1")
    if target not in _TARGET_LABELS:
        raise ValueError(f"unsupported Kumo target: {target}")
    feature_columns = available_feature_columns(context.columns)
    context = context.select(*feature_columns, target)
    _validate_labels(context, target)
    if context[target].n_unique() < 2:
        raise ValueError(
            f"training data must contain at least two classes for {target}"
        )

    model_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    table = _table(context, model_device)
    model = KumoTabular(task="classification", size="small", device=model_device)
    object.__setattr__(model, "feature_columns", feature_columns)
    model.fit(
        x=table.drop_columns(target),
        y=table[target],
        num_estimators=num_estimators,
        generator=torch.Generator(device=table.device).manual_seed(seed % (1 << 64)),
    )
    return model


def save_kumo_context(
    context: pl.DataFrame,
    path: Path,
    *,
    num_estimators: int,
    seed: int,
    target: str = TARGET,
) -> None:
    """Save labeled context and the settings needed to refit Kumo."""
    settings = json.dumps(
        {"num_estimators": num_estimators, "seed": seed, "target": target}
    )
    with NamedTemporaryFile(dir=path.parent, delete=False) as file:
        staged = Path(file.name)
    try:
        context.select(
            *available_feature_columns(context.columns), target
        ).write_parquet(staged, metadata={_CONTEXT_METADATA: settings})
        staged.replace(path)
    finally:
        staged.unlink(missing_ok=True)


def load_kumo(path: Path, *, device: str | None = None) -> KumoTabular:
    """Rebuild a fitted classifier from the saved context and its fit settings."""
    with path.open("rb") as file:
        metadata = pl.read_parquet_metadata(file)
        if _CONTEXT_METADATA not in metadata:
            raise ValueError(f"not a saved Kumo context: {path}")
        settings = json.loads(metadata[_CONTEXT_METADATA])
        target = settings.get("target", TARGET)
        file.seek(0)
        context = pl.read_parquet(file)
    return fit_kumo(
        context,
        num_estimators=settings["num_estimators"],
        seed=settings["seed"],
        device=device,
        target=target,
    )


def predict_probabilities(model: KumoTabular, states: pl.DataFrame) -> pl.DataFrame:
    """Return one probability column per class present in the fitted context."""
    columns = getattr(model, "feature_columns", None) or available_feature_columns(
        states.columns
    )
    query = _table(states.select(columns), next(model.parameters()).device)
    return cast(pl.DataFrame, pl.from_arrow(model.predict(query).to_arrow()))


def predict_round_end_reason(model: KumoTabular, states: pl.DataFrame) -> pl.DataFrame:
    """Return one probability column per round-end reason present in the context."""
    return predict_probabilities(model, states)


def main() -> None:
    parser = ArgumentParser(
        description="Fit KumoTabular small to CS2 round-end reasons."
    )
    parser.add_argument(
        "inputs", type=Path, nargs="+", help="Training Parquet files or directories"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output context Parquet file"
    )
    parser.add_argument(
        "--device", help="PyTorch device (default: CUDA if available, otherwise CPU)"
    )
    parser.add_argument(
        "--num-estimators",
        type=int,
        default=DEFAULT_ESTIMATORS,
        help="Ensemble size (default: %(default)s)",
    )
    parser.add_argument(
        "--context-rows",
        type=int,
        default=DEFAULT_CONTEXT_ROWS,
        help="Maximum context rows (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Sampling and fitting seed (default: %(default)s)",
    )
    args = parser.parse_args()
    if args.num_estimators < 1:
        parser.error("--num-estimators must be at least 1")

    try:
        paths = _parquet_paths(args.inputs, args.output)
        context = sample_context(paths, max_rows=args.context_rows, seed=args.seed)
        model = fit_kumo(
            context,
            num_estimators=args.num_estimators,
            seed=args.seed,
            device=args.device,
        )
    except ModuleNotFoundError as error:
        if error.name in {"sdm", "torch"}:
            parser.error("run with `uv run --group kumo cs-win-prob-train-kumo`")
        raise
    except (ValueError, OSError, pl.exceptions.PolarsError) as error:
        parser.error(str(error))

    save_kumo_context(
        context, args.output, num_estimators=args.num_estimators, seed=args.seed
    )
    print(
        f"Fit KumoTabular small on {next(model.parameters()).device} with "
        f"{args.num_estimators} estimators and {context.height} context rows from "
        f"{len(paths)} Parquet files. Saved {args.output}"
    )
