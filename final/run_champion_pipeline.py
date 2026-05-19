#!/usr/bin/env python3
"""Single-entry deployment runner for final OCR pipeline.

This wrapper intentionally runs only one champion bundle.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

CHAMPION_BUNDLE = (
    "omega2_fixed="
    "data_input/H2,preprocess/H1,ocr/H4,parsers/H2,parsers/H4,"
    "qr_barcode/H1,track_merge/H1"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run fixed champion OCR pipeline from final package")
    p.add_argument("--final-root", default=".", help="Path to final package root")
    p.add_argument("--output-root", default="./repro_outputs", help="Output directory")
    p.add_argument("--mode", choices=["sample", "full"], default="sample")
    p.add_argument("--sample-size", type=int, default=96)
    p.add_argument("--visual-panel-size", type=int, default=24)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--timeout", type=int, default=-1)
    p.add_argument("--jupyter-cmd", default="jupyter")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    final_root = Path(args.final_root).resolve()

    required = [
        final_root / "hypothesis_campaign.py",
        final_root / "notebookc9d692d630.ipynb",
        final_root / "products_v2_merged.csv",
        final_root / "lenta_tech_life_hack_text.md",
        final_root / "top_crops",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("Missing required files/directories:")
        for m in missing:
            print(" -", m)
        return 2

    cmd = [
        sys.executable,
        "hypothesis_campaign.py",
        "run_bundle",
        "--project-root",
        ".",
        "--dataset-root",
        "./top_crops",
        "--task-path",
        "./lenta_tech_life_hack_text.md",
        "--notebook",
        "notebookc9d692d630.ipynb",
        "--output-root",
        args.output_root,
        "--mode",
        args.mode,
        "--sample-size",
        str(args.sample_size),
        "--visual-panel-size",
        str(args.visual_panel_size),
        "--seed",
        str(args.seed),
        "--timeout",
        str(args.timeout),
        "--jupyter-cmd",
        args.jupyter_cmd,
        "--products-dict-csv",
        "./products_v2_merged.csv",
        "--bundle",
        CHAMPION_BUNDLE,
    ]

    if (final_root / "google_dict_normalized.csv").exists():
        cmd.extend(["--google-dict-csv", "./google_dict_normalized.csv"])

    print("Running fixed champion pipeline")
    print("Final root:", final_root)
    print("Command:", " ".join(cmd))

    return subprocess.call(cmd, cwd=str(final_root))


if __name__ == "__main__":
    raise SystemExit(main())
