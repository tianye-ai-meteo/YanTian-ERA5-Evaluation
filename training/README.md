# Training Code

This directory contains the 1-degree YanTian training pipeline.

## Layout

```text
training/
|-- pretraining/      # 6-hour single-step pretraining
|-- finetuning/       # Multi-day autoregressive fine-tuning
`-- data_process/     # ERA5 NetCDF to normalized 1-degree npy
```

## Data Format

Training scripts expect normalized `.npy` files:

```text
{YANTIAN_TRAIN_DATA_DIR}/YYYY/YYYYMMDD/ERA5_Global_LM_YYYYMMDDHH.npy
```

Each file has shape `(69, 180, 360)` and is already normalized with the
statistics in `data_process/statistics.json`.

## Data Preparation

```bash
cd training/data_process
export YANTIAN_ERA5_NC_ROOT=/path/to/ERA5-Global-LM
export YANTIAN_ERA5_NORM_OUTPUT_ROOT=/path/to/ERA5-Global-LM-1-norm
python data_Global_LM-1_norm.py
```

## Pretraining

```bash
cd training/pretraining
export YANTIAN_TRAIN_DATA_DIR=/path/to/ERA5-Global-LM-1-norm
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
  --master_addr=127.0.0.1 --master_port=29501 train.py
```

## Fine-tuning

```bash
cd training/finetuning
export YANTIAN_TRAIN_DATA_DIR=/path/to/ERA5-Global-LM-1-norm
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
  --master_addr=127.0.0.1 --master_port=29500 trainddpfp16.py
```

## Configuration

The original defaults are preserved, but common paths and distributed-training
settings can be overridden through environment variables:

| Variable | Purpose |
|----------|---------|
| `YANTIAN_TRAIN_DATA_DIR` | Normalized training data root |
| `YANTIAN_TRAIN_WORLD_SIZE` | Expected DDP world size |
| `YANTIAN_TRAIN_TOTAL_BATCH_SIZE` | Global training batch size |
| `YANTIAN_NUM_WORKERS` | DataLoader workers |
| `YANTIAN_PRETRAIN_CHECKPOINT_DIR` | Pretraining checkpoint output |
| `YANTIAN_FINETUNE_CHECKPOINT_DIR` | Fine-tuning checkpoint output |
| `YANTIAN_PRETRAIN_CONTINUE_TRAIN` | Resume pretraining from an existing checkpoint |
| `YANTIAN_FINETUNE_CONTINUE_TRAIN` | Resume fine-tuning from an existing checkpoint |

Checkpoints, logs, and generated figures are ignored by Git.
