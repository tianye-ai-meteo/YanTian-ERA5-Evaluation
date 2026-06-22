# YanTian ERA5 Evaluation

This repository runs the YanTian forecast model on 1-degree ERA5 input data and
evaluates the day-3 forecast against ERA5.

The default experiment is:

- Start time: `2020-01-01 00:00`
- Verification time: `2020-01-04 00:00`
- Lead time: `72 h` / 12 autoregressive 6-hour forecast steps
- Grid: `1 degree`, shape `(180, 360)`
- Evaluation variables: `Z500`, `T850`, `T2M`, `U10`, `V10`, `MSLP`
- Metric: global latitude-weighted RMSE

This repository does not use GFS data and does not use the downscaler. All
inference and evaluation are performed on the raw 1-degree forecast model
output.

## Paper

**Searth Transformer: A Transformer Architecture Incorporating Earth's
Geospheric Physical Priors for Global Mid-Range Weather Forecasting**

Paper link: https://doi.org/10.48550/arXiv.2601.09467

## Model

Download the forecast ONNX model files and place them in the project root:

| File | Source |
|------|--------|
| `YanTian_forecast.onnx` | [Google Drive](https://drive.google.com/drive/folders/1DhDZR79buQYBOBTY2ini_Q4_bVY2DJ5C?usp=sharing) / [Baidu Pan](https://pan.baidu.com/s/1NpDJLqNMjlNcK8ic-ZPbzA?pwd=7dad) |
| `YanTian_forecast.data` | Same cloud drive |

The downscaler files are not required for this repository.

## ERA5 Data

ERA5 data are not stored in this GitHub repository because the files are too
large. Download the prepared 1-degree ERA5 files from:

| Source | Link |
|--------|------|
| Google Drive | [era5_1deg](https://drive.google.com/drive/folders/1Rn9L0jaymGlUVfTWD8JnjNJHMVIcbCdt?usp=sharing) |
| Baidu Pan | [era5_1deg](https://pan.baidu.com/s/1M8vUPmM0q7E-t-KP_HlMkw?pwd=xub7), extraction code: `xub7` |

The `.npy` files linked here are prepared outside this repository from ERA5
NetCDF source files. The local preprocessing step extracts the required ERA5
variables, arranges them in the YanTian 69-channel order, and averages the
0.25-degree grid to `180 x 360`. That preprocessing code and the source `.nc`
files are not tracked in GitHub.

After downloading, place files under:

```text
data/era5_1deg/
```

Required files for the default experiment:

```text
data/era5_1deg/era5_2019123118.npy
data/era5_1deg/era5_2020010100.npy
data/era5_1deg/era5_2020010400.npy
```

Each file must be a NumPy array with shape:

```text
(69, 180, 360)
```

The array must already be on the 1-degree grid produced by averaging the
original high-resolution ERA5 fields to `180 x 360`. Values must be in physical
units, not normalized values. `inference.py` normalizes the model input using
`statistics.json` and denormalizes the forecast before computing RMSE.

## Variable Order

The channel order follows the YanTian 69-variable convention.

Pressure levels:

```text
[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000] hPa
```

Channels:

| Channels | Variable |
|----------|----------|
| `0-12` | `Z` at 13 pressure levels, geopotential in `m2 s-2` |
| `13-25` | `R` at 13 pressure levels |
| `26-38` | `T` at 13 pressure levels |
| `39-51` | `U` at 13 pressure levels |
| `52-64` | `V` at 13 pressure levels |
| `65` | `U10` |
| `66` | `V10` |
| `67` | `T2M` |
| `68` | `MSLP` |

The default RMSE variables use these channels:

| Metric name | Channel |
|-------------|---------|
| `Z500` | `7` |
| `T850` | `36` |
| `T2M` | `67` |
| `U10` | `65` |
| `V10` | `66` |
| `MSLP` | `68` |

## Environment

Create the conda environment:

```bash
conda env create -f environment.txt
conda activate yantian-era5
```

Or install the minimal Python dependencies manually:

```bash
pip install numpy onnxruntime
```

Use a GPU-enabled ONNX Runtime build if you want GPU inference.

## Validate Data

Check that the default ERA5 `.npy` files exist and have the expected shape:

```bash
python prepare_data.py
```

Equivalent check through the inference script:

```bash
python inference.py --check-data-only
```

## Run Inference And RMSE

Default run:

```bash
python inference.py
```

This performs:

```text
ERA5(t-6h), ERA5(t)
  -> normalize with statistics.json
  -> YanTian_forecast.onnx
  -> autoregressive 6-hour rolling forecast for 12 steps
  -> denormalize the 2020-01-04 00:00 forecast
  -> compare with ERA5 at 2020-01-04 00:00
  -> compute global latitude-weighted RMSE
```

Outputs:

```text
predict/forecast_2020010100_to_2020010400_1deg.npy
predict/rmse_2020010100_to_2020010400_1deg.json
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

To save every 6-hour forecast step as well:

```bash
python inference.py --save-all-steps
```

To run a different start and verification time:

```bash
python inference.py \
  --start-time 2020010100 \
  --target-time 2020010400 \
  --data-dir data/era5_1deg \
  --output-dir predict
```

## RMSE Definition

The RMSE is computed for each selected variable on the 1-degree grid:

```text
RMSE = sqrt(weighted_mean((forecast - ERA5)^2))
```

Latitude weights are proportional to `cos(latitude)`. The longitude dimension is
averaged uniformly.

## Project Structure

```text
YanTian-ERA5-Evaluation/
|-- inference.py              # ERA5 1-degree inference and RMSE evaluation
|-- prepare_data.py           # ERA5 npy shape checker
|-- statistics.json           # 69-channel normalization statistics
|-- environment.txt           # Minimal runtime environment
|-- training/                 # Training code placeholder; to be filled later
|-- YanTian_forecast.onnx     # External model file, not tracked by Git
|-- YanTian_forecast.data     # External model data file, not tracked by Git
|-- data/era5_1deg/           # External ERA5 npy files, not tracked by Git
|-- predict/                  # Output directory, not tracked by Git
`-- README.md
```

## Training Code

Training code will be added under `training/` after the source location is
confirmed. The current repository is focused on ERA5 inference and evaluation.

## Citation

```bibtex
@article{li2025searth,
  title={Searth Transformer: A Transformer Architecture Incorporating Earth's Geospheric Physical Priors for Global Mid-Range Weather Forecasting},
  author={Li, Tianye and others},
  journal={arXiv preprint arXiv:2601.09467},
  year={2025}
}
```

---

# YanTian ERA5 测评仓库

本仓库用于在 1 度 ERA5 输入数据上运行 YanTian 预报模型，并计算第三天重点变量
相对于 ERA5 的全球纬度加权 RMSE。

默认实验设置：

- 起报时间：`2020-01-01 00:00`
- 验证时间：`2020-01-04 00:00`
- 预报时效：`72 h`，即 12 个 6 小时自回归步长
- 网格：`1 degree`，维度 `(180, 360)`
- 评估变量：`Z500`、`T850`、`T2M`、`U10`、`V10`、`MSLP`
- 指标：全球纬度加权 RMSE

本仓库不使用 GFS 数据，也不使用降尺度模型；所有推理和评估都在 forecast model 的
1 度原始输出上完成。

## 模型文件

下载 forecast ONNX 模型文件，并放在项目根目录：

| 文件 | 下载方式 |
|------|----------|
| `YanTian_forecast.onnx` | [Google Drive](https://drive.google.com/drive/folders/1DhDZR79buQYBOBTY2ini_Q4_bVY2DJ5C?usp=sharing) / [百度网盘](https://pan.baidu.com/s/1NpDJLqNMjlNcK8ic-ZPbzA?pwd=7dad)，提取码：`7dad` |
| `YanTian_forecast.data` | 同上 |

本仓库不需要下载降尺度模型文件。

## ERA5 数据

ERA5 数据文件过大，不上传到 GitHub。请从下方网盘链接下载已经处理好的 1 度
`.npy` 文件：

| 来源 | 链接 |
|------|------|
| Google Drive | [era5_1deg](https://drive.google.com/drive/folders/1Rn9L0jaymGlUVfTWD8JnjNJHMVIcbCdt?usp=sharing) |
| 百度网盘 | [era5_1deg](https://pan.baidu.com/s/1M8vUPmM0q7E-t-KP_HlMkw?pwd=xub7)，提取码：`xub7` |

这些 `.npy` 文件由本地 ERA5 NetCDF 源文件预处理得到。预处理过程包括：提取所需
ERA5 变量、按 YanTian 69 通道顺序组装、将 0.25 度网格通过平均法处理为
`180 x 360`。该预处理代码和源 `.nc` 文件不放入 GitHub 仓库。

下载后放到：

```text
data/era5_1deg/
```

默认实验需要：

```text
data/era5_1deg/era5_2019123118.npy
data/era5_1deg/era5_2020010100.npy
data/era5_1deg/era5_2020010400.npy
```

每个文件必须是物理量原值，不是归一化值，形状必须为：

```text
(69, 180, 360)
```

`inference.py` 会在推理前使用 `statistics.json` 对输入做归一化，并在计算 RMSE 前
将 forecast model 输出反归一化回物理量。

运行：

```bash
conda env create -f environment.txt
conda activate yantian-era5
python prepare_data.py
python inference.py
```

输出文件：

```text
predict/forecast_2020010100_to_2020010400_1deg.npy
predict/rmse_2020010100_to_2020010400_1deg.json
```

默认实验已验证得到的纬度加权 RMSE：

| 变量 | RMSE |
|------|------|
| `Z500` | `144.209382` |
| `T850` | `1.132371` |
| `T2M` | `1.056012` |
| `U10` | `1.571451` |
| `V10` | `1.611979` |
| `MSLP` | `161.366700` |

## 训练代码

训练代码将在确认源代码地址后加入 `training/` 目录。当前仓库先完成 ERA5 推理和测评流程。
