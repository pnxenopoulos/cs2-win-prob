from argparse import ArgumentParser
from pathlib import Path

from awpy import Demo

from cs_win_prob.data_preparation import (
    DEFAULT_TICKS_PER_ROUND,
    demo_metadata,
    prepare_demo,
    sha256_file,
)


def main() -> None:
    parser = ArgumentParser(
        description="Sample labeled CS2 game states into a Parquet file."
    )
    parser.add_argument("demo", type=Path, help="CS2 demo file")
    parser.add_argument("output", type=Path, help="Output Parquet file")
    parser.add_argument(
        "--ticks-per-round",
        type=int,
        default=DEFAULT_TICKS_PER_ROUND,
        help="Number of distinct samples per round (default: %(default)s)",
    )
    parser.add_argument("--seed", type=int, help="Random seed for repeatable sampling")
    args = parser.parse_args()
    if args.ticks_per_round < 1:
        parser.error("--ticks-per-round must be at least 1")
    if args.demo.resolve() == args.output.resolve():
        parser.error("the output must be a different file from the input demo")

    prepare_demo(
        Demo(args.demo),
        ticks_per_round=args.ticks_per_round,
        seed=args.seed,
        include_tick=True,
    ).write_parquet(
        args.output, metadata=demo_metadata(args.demo, sha256_file(args.demo))
    )
