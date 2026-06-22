"""Check whether the YanTian ERA5 evaluation repository is ready to run."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "era5_1deg"
REQUIRED_TIMES = ("2019123118", "2020010100", "2020010400")
EVALUATION_PACKAGES = (
    "numpy",
    "onnxruntime",
)
TRAINING_PACKAGES = (
    "xarray",
    "netCDF4",
    "matplotlib",
    "timm",
    "einops",
)


def check_packages(packages: tuple[str, ...]) -> list[str]:
    missing = []
    for package in packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing.append(package)
    return missing


def check_files(data_dir: Path) -> list[str]:
    missing = []
    for path in [ROOT / "YanTian_forecast.onnx", ROOT / "YanTian_forecast.data"]:
        if not path.exists():
            missing.append(str(path))

    for date_time in REQUIRED_TIMES:
        path = data_dir / f"era5_{date_time}.npy"
        if not path.exists():
            missing.append(str(path))
    return missing


def check_data_shapes(data_dir: Path) -> list[str]:
    errors = []
    for date_time in REQUIRED_TIMES:
        path = data_dir / f"era5_{date_time}.npy"
        if not path.exists():
            continue
        arr = np.load(path, mmap_mode="r")
        if arr.shape != (69, 180, 360):
            errors.append(f"{path}: expected shape (69, 180, 360), found {arr.shape}")
    return errors


def check_onnx_load() -> str | None:
    model_path = ROOT / "YanTian_forecast.onnx"
    if not model_path.exists():
        return None

    import onnxruntime as ort

    try:
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        input_shape = session.get_inputs()[0].shape
        output_shape = session.get_outputs()[0].shape
    except Exception as exc:
        return f"ONNX Runtime could not load {model_path}: {exc}"

    if len(input_shape) != 5:
        return f"Unexpected ONNX input rank: {input_shape}"
    if len(output_shape) != 4:
        return f"Unexpected ONNX output rank: {output_shape}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Check YanTian ERA5 evaluation installation.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--skip-onnx-load", action="store_true")
    parser.add_argument("--include-training", action="store_true", help="Also check training/preprocessing packages.")
    args = parser.parse_args()

    failures = []

    packages = EVALUATION_PACKAGES + (TRAINING_PACKAGES if args.include_training else ())
    missing_packages = check_packages(packages)
    if missing_packages:
        failures.append("Missing Python packages: " + ", ".join(missing_packages))

    missing_files = check_files(args.data_dir)
    if missing_files:
        failures.append("Missing required files:\n  " + "\n  ".join(missing_files))

    shape_errors = check_data_shapes(args.data_dir)
    if shape_errors:
        failures.append("Invalid ERA5 data files:\n  " + "\n  ".join(shape_errors))

    if not args.skip_onnx_load:
        onnx_error = check_onnx_load()
        if onnx_error:
            failures.append(onnx_error)

    if failures:
        print("Installation check failed.\n")
        for item in failures:
            print(f"- {item}")
        return 1

    print("Installation check passed.")
    print(f"Repository: {ROOT}")
    print(f"Data directory: {args.data_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
