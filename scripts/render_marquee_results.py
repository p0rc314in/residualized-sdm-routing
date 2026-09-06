#!/usr/bin/env python3
"""Render the BabyLM data into its README section."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared_residual_routing.results import (  # noqa: E402
    load_json,
    render_marquee,
    replace_marquee,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    result_path = ROOT / "data/marquee-results.json"
    readme_path = ROOT / "README.md"
    expected = replace_marquee(
        readme_path.read_text(),
        render_marquee(load_json(result_path)),
    )
    if args.check:
        if expected != readme_path.read_text():
            raise SystemExit("README BabyLM result section is out of date")
        print("BabyLM result section is current")
        return
    readme_path.write_text(expected)
    print(f"updated {readme_path}")


if __name__ == "__main__":
    main()
