from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal

from cs_win_prob.data_preparation import FEATURE_COLUMNS
from cs_win_prob.kumo import WINNER_TARGET, sample_context


def test_context_retains_rare_classes_across_parquet_batches(tmp_path: Path) -> None:
    first = tmp_path / "a.parquet"
    second = tmp_path / "b.parquet"
    rows = pl.DataFrame(
        {
            "map": ["de_mirage"] * 20_000,
            "phase": ["bomb_not_planted"] * 20_000,
            "round_winner": ["CT"] * 20_000,
            "round_end_reason": ["t_eliminated"] * 19_999 + ["bomb_defusal"],
        }
    ).with_columns(
        pl.int_range(pl.len()).alias("time_since_phase_start"),
        *(pl.lit(0).alias(name) for name in FEATURE_COLUMNS[3:]),
    )
    rows.write_parquet(first)
    rows.with_columns(
        pl.lit("T").alias("round_winner"),
        pl.lit("bomb_exploded").alias("round_end_reason"),
    ).write_parquet(second)

    context = sample_context([first, second], max_rows=10, seed=42)

    assert context.height == 10
    assert context.columns == [*FEATURE_COLUMNS, "round_end_reason"]
    assert set(context["round_end_reason"]) == {
        "t_eliminated",
        "bomb_defusal",
        "bomb_exploded",
    }
    assert_frame_equal(context, sample_context([first, second], max_rows=10, seed=42))

    winners = sample_context([first, second], max_rows=2, seed=42, target=WINNER_TARGET)
    assert winners.columns == [*FEATURE_COLUMNS, WINNER_TARGET]
    assert set(winners[WINNER_TARGET]) == {"CT", "T"}
