# Fine-tuning

This folder contains the multi-day autoregressive fine-tuning entry point:

```bash
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
  --master_addr=127.0.0.1 --master_port=29500 trainddpfp16.py
```

Set `YANTIAN_TRAIN_DATA_DIR` to the normalized 1-degree ERA5 training data root
before launching. Additional runtime paths and batch settings can be overridden
through the environment variables documented in `../README.md`.
