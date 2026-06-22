# Data Availability

This repository does not track large model weights, ERA5 evaluation files,
training datasets, checkpoints, or prediction outputs.

## Files Required for the Default Evaluation

Download the following external files before running the default evaluation:

```text
YanTian_forecast.onnx
YanTian_forecast.data
data/era5_1deg/era5_2019123118.npy
data/era5_1deg/era5_2020010100.npy
data/era5_1deg/era5_2020010400.npy
```

Model files:

| Source | Link |
|--------|------|
| Google Drive | https://drive.google.com/drive/folders/1DhDZR79buQYBOBTY2ini_Q4_bVY2DJ5C?usp=sharing |
| Baidu Pan | https://pan.baidu.com/s/1NpDJLqNMjlNcK8ic-ZPbzA?pwd=7dad, code `7dad` |

Prepared ERA5 1-degree evaluation files:

| Source | Link |
|--------|------|
| Google Drive | https://drive.google.com/drive/folders/1Rn9L0jaymGlUVfTWD8JnjNJHMVIcbCdt?usp=sharing |
| Baidu Pan | https://pan.baidu.com/s/1M8vUPmM0q7E-t-KP_HlMkw?pwd=xub7, code `xub7` |

## Recommended Archival Repository

For long-term reuse, the model files and prepared ERA5 evaluation files should
be deposited in a persistent research-data repository such as Mendeley Data,
Zenodo, or Figshare. After receiving a DOI, replace the temporary cloud drive
links in `README.md` and this file with the DOI link.

## Raw ERA5 Source

ERA5 is provided by the Copernicus Climate Data Store. The files used here are
derived products prepared from ERA5 NetCDF source files by selecting the YanTian
69-channel variable set and averaging the native 0.25-degree grid to a
1-degree `(180, 360)` grid.

## Data Statement Template

```text
The source code is available at https://github.com/tianye-ai-meteo/YanTian-ERA5-Evaluation under the MIT License. Model files and prepared ERA5 1-degree evaluation data are available at [DATA DOI TO BE ADDED]. ERA5 source data are available from the Copernicus Climate Data Store.
```
