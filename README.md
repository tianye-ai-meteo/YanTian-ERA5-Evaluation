# YanTian ERA5 Evaluation

This repository contains a 1-degree ERA5 evaluation workflow for the YanTian v1
forecast model, plus the training code used for 1-degree pretraining and
multi-day autoregressive fine-tuning.

The default evaluation runs the forecast model from `2020-01-01 00:00` to
`2020-01-04 00:00` and reports global latitude-weighted RMSE for `Z500`,
`T850`, `T2M`, `U10`, `V10`, and `MSLP`.

## Repository Scope

- Inference and RMSE evaluation on ERA5 1-degree data.
- Data validation for prepared ERA5 `.npy` files.
- Training code for 1-degree pretraining and fine-tuning.
- Local preprocessing code for converting raw ERA5 NetCDF files into normalized
  1-degree training `.npy` files.

Large model files, ERA5 data, checkpoints, logs, and prediction outputs are not
tracked by Git.

## License

This software is released under the MIT License. See `LICENSE`.

## Project Layout

```text
YanTian-ERA5-Evaluation/
|-- inference.py                    # ERA5 1-degree inference and RMSE evaluation
|-- prepare_data.py                 # ERA5 evaluation npy shape checker
|-- statistics.json                 # 69-channel normalization statistics
|-- environment.txt                 # Runtime environment skeleton
|-- requirements.txt                # Minimal Python dependencies
|-- training/
|   |-- README.md                   # Training workflow notes
|   |-- pretraining/                # 6-hour forecast pretraining code
|   |-- finetuning/                 # Multi-day autoregressive fine-tuning code
|   `-- data_process/               # ERA5 NetCDF to normalized 1-degree npy
|-- data/era5_1deg/                 # External evaluation data, ignored by Git
|-- predict/                        # Inference outputs, ignored by Git
|-- YanTian_forecast.onnx           # External model file, ignored by Git
`-- YanTian_forecast.data           # External model data file, ignored by Git
```

## Model Files

Download the forecast ONNX files and place them in the repository root:

| File | Source |
|------|--------|
| `YanTian_forecast.onnx` | [Google Drive](https://drive.google.com/drive/folders/1DhDZR79buQYBOBTY2ini_Q4_bVY2DJ5C?usp=sharing) / [Baidu Pan](https://pan.baidu.com/s/1NpDJLqNMjlNcK8ic-ZPbzA?pwd=7dad), code `7dad` |
| `YanTian_forecast.data` | Same cloud drive |

The downscaler is not used in this repository.

## ERA5 Evaluation Data

Prepared ERA5 1-degree `.npy` files are distributed outside GitHub:

| Source | Link |
|--------|------|
| Google Drive | [era5_1deg](https://drive.google.com/drive/folders/1Rn9L0jaymGlUVfTWD8JnjNJHMVIcbCdt?usp=sharing) |
| Baidu Pan | [era5_1deg](https://pan.baidu.com/s/1M8vUPmM0q7E-t-KP_HlMkw?pwd=xub7), code `xub7` |

Place the files under:

```text
data/era5_1deg/
```

Required files for the default evaluation:

```text
data/era5_1deg/era5_2019123118.npy
data/era5_1deg/era5_2020010100.npy
data/era5_1deg/era5_2020010400.npy
```

Each file must have shape `(69, 180, 360)` and contain physical values, not
normalized values. `inference.py` normalizes the input with `statistics.json`,
then denormalizes the forecast before computing RMSE.

## Variable Order

The 69-channel order is:

| Channels | Variable |
|----------|----------|
| `0-12` | `Z` at `[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]` hPa |
| `13-25` | `R` at the same 13 pressure levels |
| `26-38` | `T` at the same 13 pressure levels |
| `39-51` | `U` at the same 13 pressure levels |
| `52-64` | `V` at the same 13 pressure levels |
| `65` | `U10` |
| `66` | `V10` |
| `67` | `T2M` |
| `68` | `MSLP` |

Default RMSE channels:

| Metric | Channel |
|--------|---------|
| `Z500` | `7` |
| `T850` | `36` |
| `T2M` | `67` |
| `U10` | `65` |
| `V10` | `66` |
| `MSLP` | `68` |

## Environment

Create the base environment with conda:

```bash
conda env create -f environment.txt
conda activate yantian-era5
```

Alternatively, install the Python package dependencies with pip:

```bash
python -m pip install -r requirements.txt
```

For training, install PyTorch with the CUDA build matching your machine before
launching DDP training jobs.

## Quick Start

After installing dependencies and downloading the external model/data files,
run:

```bash
python prepare_data.py
python inference.py
```

This should create:

```text
predict/forecast_2020010100_to_2020010400_1deg.npy
predict/rmse_2020010100_to_2020010400_1deg.json
```

To check file availability without loading the ONNX model:

```bash
python inference.py --check-data-only
```

Validated default-run RMSE:

| Variable | Latitude-weighted RMSE |
|----------|------------------------|
| `Z500` | `144.209382` |
| `T850` | `1.132371` |
| `T2M` | `1.056012` |
| `U10` | `1.571451` |
| `V10` | `1.611979` |
| `MSLP` | `161.366700` |

## Run a Different Evaluation

```bash
python inference.py \
  --start-time 2020010100 \
  --target-time 2020010400 \
  --data-dir data/era5_1deg \
  --output-dir predict
```

## Training Data Preparation

The training pipeline expects normalized 1-degree ERA5 files named:

```text
ERA5_Global_LM_YYYYMMDDHH.npy
```

with this directory layout:

```text
{YANTIAN_TRAIN_DATA_DIR}/YYYY/YYYYMMDD/ERA5_Global_LM_YYYYMMDDHH.npy
```

To generate normalized training files from raw ERA5 NetCDF files:

```bash
cd training/data_process
export YANTIAN_ERA5_NC_ROOT=/path/to/ERA5-Global-LM
export YANTIAN_ERA5_NORM_OUTPUT_ROOT=/path/to/ERA5-Global-LM-1-norm
python data_Global_LM-1_norm.py
```

The preprocessing script reads `training/data_process/statistics.json`, extracts
the 69 YanTian channels, averages the 0.25-degree grid to `(180, 360)`,
normalizes by channel, and saves `float16` `.npy` files.

## Training

Set the normalized training data root:

```bash
export YANTIAN_TRAIN_DATA_DIR=/path/to/ERA5-Global-LM-1-norm
```

Pretraining:

```bash
cd training/pretraining
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
  --master_addr=127.0.0.1 --master_port=29501 train.py
```

Fine-tuning:

```bash
cd training/finetuning
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
  --master_addr=127.0.0.1 --master_port=29500 trainddpfp16.py
```

Useful environment overrides:

| Variable | Purpose |
|----------|---------|
| `YANTIAN_TRAIN_DATA_DIR` | Normalized training data root |
| `YANTIAN_TRAIN_WORLD_SIZE` | Expected number of DDP processes |
| `YANTIAN_TRAIN_TOTAL_BATCH_SIZE` | Global batch size |
| `YANTIAN_NUM_WORKERS` | DataLoader workers |
| `YANTIAN_PRETRAIN_CHECKPOINT_DIR` | Pretraining checkpoint output |
| `YANTIAN_FINETUNE_CHECKPOINT_DIR` | Fine-tuning checkpoint output |
| `YANTIAN_PRETRAIN_CONTINUE_TRAIN` | Resume pretraining from an existing checkpoint |
| `YANTIAN_FINETUNE_CONTINUE_TRAIN` | Resume fine-tuning from an existing checkpoint |

## Notes

- The training scripts are GPU/DDP-oriented and expect `torchrun`.
- Fine-tuning defaults to checkpoint resume mode. Set
  `YANTIAN_FINETUNE_CONTINUE_TRAIN=false` only when intentionally starting
  fine-tuning without an existing checkpoint.
- Checkpoints and training logs are ignored by Git.
- The evaluation workflow uses the forecast model only; it does not use GFS
  input data or the downscaler.
