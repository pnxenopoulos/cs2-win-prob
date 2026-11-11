"""Plot predicted round win probabilities from a held-out demo."""

from __future__ import annotations

from argparse import ArgumentParser
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from awpy import Demo

from cs_win_prob.data_preparation import prepare_round_timeline
from cs_win_prob.kumo import load_kumo, predict_probabilities

if TYPE_CHECKING:
    from sdm.models import KumoTabular


def score_rounds(model: KumoTabular, demo: Demo, *, rounds: int = 10) -> pl.DataFrame:
    """Score live ticks from the first rounds with a fitted round-winner model."""
    states = prepare_round_timeline(demo, rounds=rounds)
    probabilities = pl.concat(
        [
            predict_probabilities(model, batch)
            for batch in states.iter_slices(n_rows=256)
        ]
    )
    if set(probabilities.columns) != {"CT", "T"}:
        raise ValueError("expected a Kumo context fitted to round_winner")
    return states.select(
        "round_num",
        "tick",
        "round_elapsed_seconds",
        "round_duration_seconds",
        "plant_seconds",
        "round_winner",
    ).with_columns(
        probabilities["CT"].alias("ct_win_probability"),
        probabilities["T"].alias("t_win_probability"),
    )


def plot_rounds(scores: pl.DataFrame, output: Path, *, title: str) -> None:
    """Render one panel per round on common probability and time scales."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator, PercentFormatter

    rounds = scores.partition_by("round_num", maintain_order=True)
    row_count = ceil(len(rounds) / 2)
    paper, ink, muted = "#f7f8fa", "#17263a", "#516276"
    blue, amber = "#1768ac", "#a76100"
    figure = Figure(figsize=(13.5, 2.05 * row_count + 1.65), facecolor=paper)
    FigureCanvasAgg(figure)
    axes = figure.subplots(
        row_count, 2, sharex=True, sharey=True, squeeze=False
    ).ravel()
    max_seconds = max(float(frame["round_duration_seconds"][0]) for frame in rounds)
    xmax = max(15, 15 * ceil(max_seconds / 15))

    for index, (axis, frame) in enumerate(
        zip(axes[: len(rounds)], rounds, strict=True)
    ):
        seconds = frame["round_elapsed_seconds"].to_numpy()
        probability = frame["ct_win_probability"].to_numpy()
        plant = frame["plant_seconds"][0]
        axis.set_facecolor("#ffffff")
        axis.set_title(
            f"Round {frame['round_num'][0]}  ·  {frame['round_winner'][0]} won",
            loc="left",
            fontsize=11,
            color=ink,
            fontweight="semibold",
            pad=7,
        )
        axis.fill_between(seconds, probability, step="post", color=blue, alpha=0.10)
        axis.plot(
            seconds, probability, drawstyle="steps-post", color=blue, linewidth=1.6
        )
        if plant is not None:
            axis.axvline(plant, color=amber, linestyle="--", linewidth=1.25)
        axis.axhline(0.5, color="#99a7b6", linestyle=":", linewidth=1)
        axis.set_xlim(0, xmax)
        axis.set_ylim(0, 1)
        axis.xaxis.set_major_locator(MultipleLocator(30))
        axis.yaxis.set_major_locator(MultipleLocator(0.25))
        axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        axis.grid(axis="y", color="#e2e7ed", linewidth=0.65)
        axis.set_axisbelow(True)
        axis.tick_params(axis="both", colors=muted, labelsize=9, length=0, pad=5)
        axis.tick_params(axis="x", labelbottom=index + 2 >= len(rounds))
        for spine in axis.spines.values():
            spine.set_visible(False)
    for axis in axes[len(rounds) :]:
        axis.set_visible(False)

    figure.subplots_adjust(
        left=0.105, right=0.975, top=0.875, bottom=0.095, hspace=0.60, wspace=0.11
    )
    figure.suptitle(
        title, x=0.105, y=0.972, ha="left", fontsize=23, fontweight="bold", color=ink
    )
    figure.text(
        0.105,
        0.918,
        f"CT round win probability · first {len(rounds)} rounds",
        fontsize=12,
        color=muted,
    )
    figure.legend(
        handles=[
            Line2D([], [], color=blue, linewidth=2, label="CT win probability"),
            Line2D([], [], color=amber, linestyle="--", label="Bomb planted"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.98, 0.947),
        frameon=False,
        fontsize=10,
        labelcolor=ink,
    )
    figure.supxlabel(
        "Seconds since freeze time ended", y=0.037, fontsize=11, color=muted
    )
    figure.supylabel("Predicted probability", x=0.024, fontsize=11, color=muted)
    figure.savefig(output, dpi=180, facecolor=paper)


def main() -> None:
    parser = ArgumentParser(
        description="Plot win probabilities for the first rounds of a CS2 demo."
    )
    parser.add_argument("demo", type=Path, help="Demo to score")
    parser.add_argument("context", type=Path, help="Kumo round-win context Parquet")
    parser.add_argument("--output", type=Path, required=True, help="Output plot image")
    parser.add_argument(
        "--predictions", type=Path, help="Also save scored ticks to Parquet"
    )
    parser.add_argument(
        "--rounds", type=int, default=10, help="Rounds to plot (default: %(default)s)"
    )
    parser.add_argument("--title", help="Figure title (defaults to the demo name)")
    parser.add_argument(
        "--device", help="PyTorch device (default: CUDA if available, otherwise CPU)"
    )
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be at least 1")
    paths = [args.demo, args.context, args.output]
    if args.predictions is not None:
        paths.append(args.predictions)
    if len({path.resolve() for path in paths}) != len(paths):
        parser.error("the inputs and outputs must be different files")

    model = load_kumo(args.context, device=args.device)
    scores = score_rounds(model, Demo(args.demo), rounds=args.rounds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_rounds(scores, args.output, title=args.title or args.demo.stem)
    if args.predictions is not None:
        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        scores.write_parquet(args.predictions)
    print(
        f"Saved {args.output} ({scores['round_num'].n_unique()} rounds, {scores.height} scored ticks)"
    )
