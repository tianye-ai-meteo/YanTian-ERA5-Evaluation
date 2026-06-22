# Project Status

## Repository Target

- GitHub account: `tianye-ai-meteo`
- Planned repository: `YanTian-ERA5-Evaluation`
- Planned remote URL: `https://github.com/tianye-ai-meteo/YanTian-ERA5-Evaluation.git`
- Local path: `/Users/litianye/Desktop/code-lab/YanTian-ERA5-Evaluation`

## Current Local Build

- ERA5 1-degree inference and evaluation workflow is available.
- Default evaluation target is `2020-01-04 00:00` from start time `2020-01-01 00:00`.
- RMSE variables: `Z500`, `T850`, `T2M`, `U10`, `V10`, `MSLP`.
- Metric: global latitude-weighted RMSE on the raw 1-degree forecast output.
- Local ERA5 `.npy` files, model files, and prediction outputs are ignored by Git.

## Pending

- Add training source code under `training/` after the source path is provided.
- Create the remote repository under `tianye-ai-meteo`.
- Push local `main` after the training code is integrated and the repository name is confirmed.
