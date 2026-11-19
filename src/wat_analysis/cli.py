"""Command-line entry point for offline WAT measurements."""

import argparse
import os
from pathlib import Path

from .wat import add_arguments, run, settings


def main():
    parser = argparse.ArgumentParser(
        prog="wat-analysis",
        description="Measure whole-anode pulses, tile recovery, and PACMAN receipt delays in 2×2 packet files.",
    )
    add_arguments(parser)
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    if os.environ.get("NERSC_HOST") and os.environ.get("SCRATCH"):
        cache_root = Path(os.environ["SCRATCH"])
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=cache_root / "2x2-wat-analysis",
        help="Directory for reusable per-file packet indexes",
    )
    args = parser.parse_args()
    try:
        run(
            args.files,
            args.cache_dir.expanduser(),
            args.output.expanduser(),
            settings(args),
        )
    except (OSError, ValueError, ImportError) as exc:
        parser.exit(2, f"WAT analysis: {exc}\n")
