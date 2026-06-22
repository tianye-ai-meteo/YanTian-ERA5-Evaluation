"""
Validate local ERA5 1-degree `.npy` files for the YanTian ERA5 evaluation project.

This script does not download data. ERA5 files are distributed separately via
cloud storage because they are too large for GitHub.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data" / "era5_1deg"
REQUIRED_TIMES = ["2019123118", "2020010100", "2020010400"]


def validate_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)

    data = np.load(path, mmap_mode="r")
    if data.shape != (69, 180, 360):
        raise ValueError(f"{path} has shape {data.shape}; expected (69, 180, 360)")
    if not np.issubdtype(data.dtype, np.number):
        raise TypeError(f"{path} has non-numeric dtype {data.dtype}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate YanTian ERA5 1-degree npy files.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--times", nargs="+", default=REQUIRED_TIMES)
    args = parser.parse_args()

    failed = False
    for date_time in args.times:
        path = args.data_dir / f"era5_{date_time}.npy"
        try:
            validate_file(path)
            data = np.load(path, mmap_mode="r")
            print(f"OK: {path} shape={data.shape} dtype={data.dtype}")
        except Exception as exc:
            failed = True
            print(f"FAILED: {path} ({exc})")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
