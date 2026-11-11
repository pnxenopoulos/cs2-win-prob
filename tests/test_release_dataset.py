from hashlib import sha256
from math import log
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cs_win_prob import release_training
from cs_win_prob.data_preparation import (
    BASE_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    demo_metadata,
)
from cs_win_prob.dataset import export_dataset


def _write_demo_states(directory: Path, number: int) -> str:
    digest = sha256(f"demo bytes {number}".encode()).hexdigest()
    states = pl.DataFrame(
        {
            "map": ["de_mirage"] * 2,
            "phase": ["bomb_not_planted"] * 2,
            "time_since_phase_start": [float(number)] * 2,
            "round_winner": ["CT", "T"],
            "round_end_reason": ["bomb_defusal", "ct_eliminated"],
        }
    ).with_columns(*(pl.lit(0).alias(name) for name in FEATURE_COLUMNS[3:]))
    states.write_parquet(
        directory / f"states-{number}.parquet",
        metadata=demo_metadata(Path(f"match-{number}.dem"), digest),
    )
    return digest


def test_export_preserves_demo_provenance_and_split(tmp_path: Path) -> None:
    states_dir = tmp_path / "states"
    states_dir.mkdir()
    expected = {
        f"match-{number}.dem": _write_demo_states(states_dir, number)
        for number in range(5)
    }
    output = tmp_path / "combined.parquet"

    assert export_dataset(states_dir, output) == (4, 1)
    first = pl.read_parquet(output)
    assert export_dataset(states_dir, output) == (4, 1)
    assert_frame_equal(first, pl.read_parquet(output))
    assert first.height == 10
    assert first.columns == [
        "demo_id",
        "demo_filename",
        "demo_sha256",
        "split",
        *FEATURE_COLUMNS,
        "round_end_reason",
    ]
    assert (
        dict(first.select("demo_filename", "demo_sha256").unique().iter_rows())
        == expected
    )
    assert first["demo_id"].equals(first["demo_sha256"])
    assert first.group_by("demo_id").agg(pl.col("split").n_unique())["split"].max() == 1
    _write_demo_states(states_dir, 5)
    expanded = tmp_path / "expanded.parquet"
    assert export_dataset(states_dir, expanded, seed=123, split_from=output) == (4, 1)
    assert_frame_equal(first, pl.read_parquet(expanded))


@pytest.mark.parametrize("target", ["round_end_reason", "round_winner"])
def test_release_fit_uses_train_only_and_scores_held_out_demos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    for number in range(2):
        _write_demo_states(tmp_path, number)
    dataset_path = tmp_path / "combined.parquet"
    export_dataset(tmp_path, dataset_path)
    dataset = pl.read_parquet(dataset_path)

    model = Mock()
    model.parameters.return_value = iter([SimpleNamespace(device="cpu")])
    fit = Mock(return_value=model)
    classes = (
        ("CT", "T") if target == "round_winner" else ("bomb_defusal", "ct_eliminated")
    )
    predict = Mock(
        return_value=pl.DataFrame({classes[0]: [0.8, 0.25], classes[1]: [0.2, 0.75]})
    )
    monkeypatch.setattr(release_training, "fit_kumo", fit)
    monkeypatch.setattr(release_training, "predict_probabilities", predict)
    monkeypatch.setattr(release_training, "version", lambda _: "test")

    report = release_training.fit_release(
        dataset_path, tmp_path / "model", baseline_features=True, target=target
    )

    fitted = fit.call_args.args[0]
    assert report["target"] == target
    assert fitted.columns == [*BASE_FEATURE_COLUMNS, target]
    assert set(fitted[target]) == set(classes)
    assert set(fitted["time_since_phase_start"]) == set(
        dataset.filter(split="train")["time_since_phase_start"]
    )
    scored = predict.call_args.args[1]
    assert set(scored["time_since_phase_start"]) == set(
        dataset.filter(split="test")["time_since_phase_start"]
    )
    assert report["test"] == pytest.approx(
        {
            "demos": 1,
            "rows": 2,
            "accuracy": 1,
            "log_loss": -(log(0.8) + log(0.75)) / 2,
            "brier_score": (0.2**2 + 0.25**2) / (2 if target == "round_winner" else 1),
        }
    )


def test_release_fit_rejects_demo_in_both_splits(tmp_path: Path) -> None:
    for number in range(2):
        _write_demo_states(tmp_path, number)
    dataset_path = tmp_path / "combined.parquet"
    export_dataset(tmp_path, dataset_path)
    dataset = pl.read_parquet(dataset_path)
    overlap = dataset.filter(split="train").with_columns(split=pl.lit("test"))
    pl.concat([dataset, overlap]).write_parquet(dataset_path)

    with pytest.raises(ValueError, match="belong to exactly one split"):
        release_training.fit_release(dataset_path, tmp_path / "model")
