"""Run experiments across multiple config files."""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.run_pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multiple experiments")
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="List of config YAML files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for config_path in args.configs:
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        run_pipeline(config_path)


if __name__ == "__main__":
    main()