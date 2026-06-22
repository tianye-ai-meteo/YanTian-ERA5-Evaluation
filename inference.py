"""
YanTian 1-degree ERA5 inference and day-3 RMSE evaluation.

This repository is intentionally ERA5-only:
  - input data are local 1-degree ERA5 `.npy` files with shape (69, 180, 360)
  - no GFS download is performed
  - no downscaler is used
  - RMSE is computed on the raw 1-degree forecast model output

Default experiment:
  start:  2020-01-01 00:00
  target: 2020-01-04 00:00, i.e. +72 h / 12 autoregressive 6-hour steps
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parent
FORECAST_ONNX = ROOT / "YanTian_forecast.onnx"
STATS_PATH = ROOT / "statistics.json"
DATA_DIR = ROOT / "data" / "era5_1deg"
OUTPUT_DIR = ROOT / "predict"

DEFAULT_START_TIME = "2020010100"
DEFAULT_TARGET_TIME = "2020010400"

PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
EVAL_VARIABLES = {
    "Z500": 7,
    "T850": 26 + PRESSURE_LEVELS.index(850),
    "T2M": 67,
    "U10": 65,
    "V10": 66,
    "MSLP": 68,
}


def parse_time(value: str) -> datetime:
    return datetime.strptime(value, "%Y%m%d%H")


def format_time(value: datetime) -> str:
    return value.strftime("%Y%m%d%H")


def required_steps(start_time: str, target_time: str) -> int:
    start_dt = parse_time(start_time)
    target_dt = parse_time(target_time)
    delta_hours = int((target_dt - start_dt).total_seconds() // 3600)
    if target_dt <= start_dt:
        raise ValueError("target_time must be later than start_time")
    if delta_hours % 6 != 0:
        raise ValueError("target_time - start_time must be divisible by 6 hours")
    return delta_hours // 6


def era5_path(data_dir: Path, date_time: str) -> Path:
    return data_dir / f"era5_{date_time}.npy"


def load_normalization_stats() -> Tuple[np.ndarray, np.ndarray]:
    """Return 69-channel YanTian normalization statistics."""
    with STATS_PATH.open("r") as f:
        stats = json.load(f)

    avg = np.asarray(stats["avg"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)

    pressure_avg = avg[7:-13]
    pressure_std = std[7:-13]
    surface_avg = np.concatenate([avg[0:2], avg[3:5]])
    surface_std = np.concatenate([std[0:2], std[3:5]])

    avg_69 = np.concatenate([pressure_avg, surface_avg]).astype(np.float32)
    std_69 = np.concatenate([pressure_std, surface_std]).astype(np.float32)
    if avg_69.shape != (69,) or std_69.shape != (69,):
        raise ValueError("statistics.json did not produce 69-channel normalization arrays")
    return avg_69, std_69


def load_era5_1deg(data_dir: Path, date_time: str) -> np.ndarray:
    """Load one ERA5 1-degree state in YanTian channel order."""
    path = era5_path(data_dir, date_time)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing ERA5 input file: {path}\n"
            "Expected files are named era5_YYYYMMDDHH.npy under data/era5_1deg/."
        )

    data = np.load(path).astype(np.float32)
    if data.shape != (69, 180, 360):
        raise ValueError(f"{path} has shape {data.shape}; expected (69, 180, 360)")
    return data


def normalize(data: np.ndarray, avg_69: np.ndarray, std_69: np.ndarray) -> np.ndarray:
    return (data - avg_69[:, None, None]) / std_69[:, None, None]


def denormalize(data: np.ndarray, avg_69: np.ndarray, std_69: np.ndarray) -> np.ndarray:
    return data * std_69[:, None, None] + avg_69[:, None, None]


def build_initial_input(data_dir: Path, start_time: str, avg_69: np.ndarray, std_69: np.ndarray) -> np.ndarray:
    """Build model input (1, 2, 69, 180, 360) from t-6h and t ERA5 states."""
    start_dt = parse_time(start_time)
    past_time = format_time(start_dt - timedelta(hours=6))

    past = normalize(load_era5_1deg(data_dir, past_time), avg_69, std_69)
    current = normalize(load_era5_1deg(data_dir, start_time), avg_69, std_69)
    return np.stack([past, current], axis=0)[None, ...].astype(np.float32)


def load_onnx_session(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing forecast model: {model_path}\n"
            "Download YanTian_forecast.onnx and YanTian_forecast.data as described in README.md."
        )

    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4

    preferred = ["CUDAExecutionProvider", "ROCMExecutionProvider"]
    available = ort.get_available_providers()
    providers = [p for p in preferred if p in available] + ["CPUExecutionProvider"]

    session = ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)
    print(f"[YanTian] Loaded {model_path.name} with {session.get_providers()[0]}")
    return session


def latitude_weights(nlat: int = 180) -> np.ndarray:
    """Cosine latitude weights for a regular global 1-degree grid."""
    lat = np.linspace(-89.5, 89.5, nlat, dtype=np.float64)
    weights = np.cos(np.deg2rad(lat))
    return weights / weights.mean()


def weighted_rmse(forecast: np.ndarray, truth: np.ndarray, channel: int) -> float:
    """Global latitude-weighted RMSE for one channel."""
    if forecast.shape != (69, 180, 360) or truth.shape != (69, 180, 360):
        raise ValueError("forecast and truth must both have shape (69, 180, 360)")

    squared_error = (forecast[channel].astype(np.float64) - truth[channel].astype(np.float64)) ** 2
    zonal_mse = squared_error.mean(axis=1)
    weights = latitude_weights(squared_error.shape[0])
    return float(np.sqrt(np.average(zonal_mse, weights=weights)))


def compute_rmse(forecast: np.ndarray, truth: np.ndarray, variables: Dict[str, int]) -> Dict[str, float]:
    return {name: weighted_rmse(forecast, truth, channel) for name, channel in variables.items()}


def run_forecast(
    start_time: str,
    target_time: str,
    data_dir: Path = DATA_DIR,
    output_dir: Path = OUTPUT_DIR,
    model_path: Path = FORECAST_ONNX,
    save_all_steps: bool = False,
) -> Tuple[Path, Path, Dict[str, float]]:
    """Run autoregressive 1-degree inference and evaluate the target time."""
    steps = required_steps(start_time, target_time)
    avg_69, std_69 = load_normalization_stats()
    session = load_onnx_session(model_path)
    input_name = session.get_inputs()[0].name

    print(f"[YanTian] Loading ERA5 initial states from {data_dir}")
    input_arr = build_initial_input(data_dir, start_time, avg_69, std_69)

    output_dir.mkdir(parents=True, exist_ok=True)
    all_steps = []
    output_1deg = None

    for step in range(steps):
        output_1deg = session.run(None, {input_name: input_arr})[0].astype(np.float32)
        if output_1deg.shape != (1, 69, 180, 360):
            raise ValueError(f"Unexpected model output shape {output_1deg.shape}; expected (1, 69, 180, 360)")

        if save_all_steps:
            all_steps.append(denormalize(output_1deg[0], avg_69, std_69))

        input_arr[:, 0, :, :, :] = input_arr[:, 1, :, :, :]
        input_arr[:, 1, :, :, :] = output_1deg
        valid_time = format_time(parse_time(start_time) + timedelta(hours=6 * (step + 1)))
        print(f"[YanTian] Step {step + 1:02d}/{steps:02d} complete: valid {valid_time}")

    if output_1deg is None:
        raise RuntimeError("No forecast was produced")

    forecast_phys = denormalize(output_1deg[0], avg_69, std_69)
    truth = load_era5_1deg(data_dir, target_time)
    rmse = compute_rmse(forecast_phys, truth, EVAL_VARIABLES)

    forecast_path = output_dir / f"forecast_{start_time}_to_{target_time}_1deg.npy"
    rmse_path = output_dir / f"rmse_{start_time}_to_{target_time}_1deg.json"
    np.save(forecast_path, forecast_phys.astype(np.float32))

    result = {
        "start_time": start_time,
        "target_time": target_time,
        "lead_time_hours": steps * 6,
        "grid": "1deg_180x360",
        "rmse": rmse,
    }
    with rmse_path.open("w") as f:
        json.dump(result, f, indent=2)

    if save_all_steps:
        steps_path = output_dir / f"forecast_steps_{start_time}_to_{target_time}_1deg.npy"
        np.save(steps_path, np.stack(all_steps, axis=0).astype(np.float32))
        print(f"[YanTian] Saved all steps: {steps_path}")

    print(f"[YanTian] Saved target forecast: {forecast_path}")
    print(f"[YanTian] Saved RMSE metrics: {rmse_path}")
    for name, value in rmse.items():
        print(f"  {name}: {value:.6g}")

    return forecast_path, rmse_path, rmse


def validate_required_files(data_dir: Path, start_time: str, target_time: str) -> Iterable[Path]:
    start_dt = parse_time(start_time)
    required_times = [
        format_time(start_dt - timedelta(hours=6)),
        start_time,
        target_time,
    ]
    return [era5_path(data_dir, date_time) for date_time in required_times]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YanTian ERA5 1-degree inference and day-3 RMSE.")
    parser.add_argument("--start-time", default=DEFAULT_START_TIME, help="Forecast start time, YYYYMMDDHH.")
    parser.add_argument("--target-time", default=DEFAULT_TARGET_TIME, help="Verification time, YYYYMMDDHH.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Directory containing era5_YYYYMMDDHH.npy files.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for forecast and RMSE outputs.")
    parser.add_argument("--model", type=Path, default=FORECAST_ONNX, help="Path to YanTian_forecast.onnx.")
    parser.add_argument("--save-all-steps", action="store_true", help="Also save all 6-hour forecast steps.")
    parser.add_argument("--check-data-only", action="store_true", help="Validate required ERA5 files and exit.")
    args = parser.parse_args()

    required_steps(args.start_time, args.target_time)
    required_files = list(validate_required_files(args.data_dir, args.start_time, args.target_time))
    if args.check_data_only:
        missing = [path for path in required_files if not path.exists()]
        for path in required_files:
            status = "OK" if path.exists() else "MISSING"
            print(f"{status}: {path}")
        if missing:
            raise SystemExit(1)
        return

    run_forecast(
        start_time=args.start_time,
        target_time=args.target_time,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_path=args.model,
        save_all_steps=args.save_all_steps,
    )


if __name__ == "__main__":
    main()
