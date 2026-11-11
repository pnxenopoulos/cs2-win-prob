"""Train a round-win classifier from HLTV RAR archives."""

import os
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from shutil import which
from subprocess import SubprocessError, run
from tempfile import NamedTemporaryFile, TemporaryDirectory

from awpy import Demo
from polars.exceptions import PolarsError

from cs_win_prob.data_preparation import (
    DEFAULT_TICKS_PER_ROUND,
    demo_metadata,
    prepare_demo,
    sha256_file,
)
from cs_win_prob.kumo import (
    DEFAULT_CONTEXT_ROWS,
    DEFAULT_ESTIMATORS,
    DEFAULT_SEED,
    WINNER_TARGET,
    fit_kumo,
    sample_context,
    save_kumo_context,
)


def _find_unrar(override: str | None) -> str:
    executable = which(override) if override else which("unrar") or which("UnRAR.exe")
    if executable:
        return executable
    if override is None:
        program_files = os.environ.get("ProgramFiles", "/mnt/c/Program Files")
        winrar = Path(program_files) / "WinRAR" / "UnRAR.exe"
        if winrar.is_file():
            return str(winrar)
    raise ValueError("UnRAR was not found; install it or pass its path with --unrar")


def _extract(executable: str, archive: Path, directory: Path) -> None:
    def windows_path(path: Path) -> str:
        return run(
            ["wslpath", "-w", str(path.resolve())],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    if os.name != "nt" and executable.lower().endswith(".exe"):
        source = windows_path(archive)
        destination = windows_path(directory) + "\\"
    else:
        source = str(archive.resolve())
        destination = str(directory.resolve()) + os.sep
    # Extract into a private directory and discard paths stored inside the archive.
    result = run(
        [
            executable,
            "e",
            "-cfg-",
            "-idq",
            "-o-",
            "-p-",
            "-y",
            source,
            "*.dem",
            destination,
        ],
        check=False,
        capture_output=True,
        text=True,
        errors="replace",
    )
    if result.returncode:
        raise RuntimeError(
            f"UnRAR exited with code {result.returncode}: {result.stderr.strip() or result.stdout.strip()}"
        )


def _prepare_archive(
    archive: Path,
    *,
    executable: str,
    work_dir: Path,
    states_dir: Path,
    excluded: tuple[str, str] | None,
    ticks_per_round: int,
    seed: int,
) -> tuple[list[Path], str]:
    with TemporaryDirectory(dir=work_dir) as extracted:
        directory = Path(extracted)
        _extract(executable, archive, directory)
        demos = sorted(
            path
            for path in directory.iterdir()
            if path.suffix.lower() == ".dem"
            and path.is_file()
            and not path.is_symlink()
        )
        if not demos:
            raise ValueError("archive contains no demo files")
        if excluded and any(path.name.casefold() == excluded[0] for path in demos):
            return [], "excluded archive containing the held-out demo name"
        digests = [(path, sha256_file(path)) for path in demos]
        if excluded and any(digest == excluded[1] for _, digest in digests):
            return [], "excluded archive containing the held-out demo"

        prepared = []
        for path, digest in digests:
            try:
                states = prepare_demo(
                    Demo(path), ticks_per_round=ticks_per_round, seed=seed
                )
            except Exception as error:
                raise RuntimeError(f"could not parse {path.name}: {error}") from error
            if states.is_empty():
                print(f"  {path.name}: no eligible rounds", flush=True)
                continue
            destination = states_dir / f"{digest}.parquet"
            with NamedTemporaryFile(dir=states_dir, delete=False) as file:
                staged = Path(file.name)
            try:
                states.write_parquet(staged, metadata=demo_metadata(path, digest))
                staged.replace(destination)
            finally:
                staged.unlink(missing_ok=True)
            prepared.append(destination)
        return prepared, f"prepared {len(prepared)} of {len(demos)} demos"


def main() -> None:
    parser = ArgumentParser(
        description="Train a Kumo round-win classifier from a directory of RAR archives."
    )
    parser.add_argument(
        "directory", type=Path, help="Directory to search recursively for .rar archives"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path, help="Output Kumo context Parquet")
    mode.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare per-demo Parquets without fitting Kumo",
    )
    parser.add_argument(
        "--exclude-demo",
        type=Path,
        help="Exclude archives containing this held-out demo",
    )
    parser.add_argument(
        "--states-dir",
        type=Path,
        help="Keep the prepared Parquet files in this directory",
    )
    parser.add_argument("--unrar", help="Path to the UnRAR executable")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Archives to process concurrently (default: %(default)s)",
    )
    parser.add_argument("--ticks-per-round", type=int, default=DEFAULT_TICKS_PER_ROUND)
    parser.add_argument("--context-rows", type=int, default=DEFAULT_CONTEXT_ROWS)
    parser.add_argument("--num-estimators", type=int, default=DEFAULT_ESTIMATORS)
    parser.add_argument(
        "--device", help="PyTorch device (default: CUDA if available, otherwise CPU)"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if not args.directory.is_dir():
        parser.error(f"expected an archive directory: {args.directory}")
    if (
        min(args.workers, args.ticks_per_round, args.num_estimators) < 1
        or args.context_rows < 2
    ):
        parser.error(
            "workers, ticks per round, and estimators must be positive, and context rows must be at least 2"
        )
    if args.prepare_only and args.states_dir is None:
        parser.error("--prepare-only requires --states-dir")
    if args.output is not None and args.output.suffix.lower() != ".parquet":
        parser.error("--output must be a .parquet file")
    archives = sorted(
        path
        for path in args.directory.rglob("*")
        if path.is_file() and path.suffix.lower() == ".rar"
    )
    if not archives:
        parser.error(f"no RAR archives found in {args.directory}")
    try:
        executable = _find_unrar(args.unrar)
        excluded = (
            (args.exclude_demo.name.casefold(), sha256_file(args.exclude_demo))
            if args.exclude_demo
            else None
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))

    work_parent = (args.output or args.states_dir).parent
    work_parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Processing {len(archives)} archives with {args.workers} workers", flush=True
    )
    with TemporaryDirectory(prefix=".demo-training-", dir=work_parent) as workspace:
        work_dir = Path(workspace)
        states_dir = args.states_dir or (work_dir / "states")
        states_dir.mkdir(parents=True, exist_ok=True)
        prepared = set()
        failures = 0
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            jobs = {
                executor.submit(
                    _prepare_archive,
                    archive,
                    executable=executable,
                    work_dir=work_dir,
                    states_dir=states_dir,
                    excluded=excluded,
                    ticks_per_round=args.ticks_per_round,
                    seed=args.seed,
                ): archive
                for archive in archives
            }
            for completed, job in enumerate(as_completed(jobs), 1):
                try:
                    paths, status = job.result()
                    prepared.update(paths)
                except (
                    OSError,
                    PolarsError,
                    RuntimeError,
                    SubprocessError,
                    ValueError,
                ) as error:
                    failures += 1
                    status = f"failed: {error}"
                print(
                    f"[{completed}/{len(archives)}] {jobs[job].name}: {status}",
                    flush=True,
                )

        if failures:
            parser.exit(
                1, f"{failures} archives failed. See the archive messages above.\n"
            )
        if not prepared:
            parser.error("no eligible demos remained after extraction and filtering")
        if args.prepare_only:
            print(f"Prepared {len(prepared)} unique demos in {states_dir}")
            return
        if args.output.resolve() in {path.resolve() for path in prepared}:
            parser.error(
                "the model output must be a different file from the prepared states"
            )

        context = sample_context(
            sorted(prepared),
            max_rows=args.context_rows,
            seed=args.seed,
            target=WINNER_TARGET,
        )
        print(
            f"Fitting on {context.height} context rows from {len(prepared)} unique demos",
            flush=True,
        )
        model = fit_kumo(
            context,
            target=WINNER_TARGET,
            num_estimators=args.num_estimators,
            seed=args.seed,
            device=args.device,
        )
        save_kumo_context(
            context,
            args.output,
            target=WINNER_TARGET,
            num_estimators=args.num_estimators,
            seed=args.seed,
        )
    print(
        f"Saved {args.output} ({next(model.parameters()).device}, {args.num_estimators} estimators)"
    )
