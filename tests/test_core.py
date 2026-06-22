from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from inference import EVAL_VARIABLES, compute_rmse, required_steps
from prepare_data import validate_file


def test_required_steps_default():
    assert required_steps("2020010100", "2020010400") == 12


def test_weighted_rmse_constant_error():
    forecast = np.ones((69, 180, 360), dtype=np.float32)
    truth = np.zeros((69, 180, 360), dtype=np.float32)
    rmse = compute_rmse(forecast, truth, EVAL_VARIABLES)
    assert set(rmse) == {"Z500", "T850", "T2M", "U10", "V10", "MSLP"}
    assert all(abs(value - 1.0) < 1e-6 for value in rmse.values())


def test_validate_file_accepts_expected_shape():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "era5_2020010100.npy"
        np.save(path, np.zeros((69, 180, 360), dtype=np.float32))
        validate_file(path)
