# AMR Demand Forecasting

This project uses synthetic warehouse data to forecast the number of transport requests expected at each station in the next 15 minutes.

## Models and tools

- **XGBoost:** main demand-forecasting model
- **STGT:** spatio-temporal graph model experiment

## Test results

On the held-out synthetic test data:

- Rows evaluated: 17,550
- MAE: 1.226 requests
- RMSE: 1.563 requests
- WAPE: 39.9%

These results are from synthetic data and may differ when tested with real AMR data.

## Data

The dataset is not included in this GitHub repository. Place the provided dataset in:

`data/AMR_FINAL_30DAY_RESEARCH_READY/`

The scripts expect files such as `station_state_5min.csv` in that folder.

## Setup

Install the Python packages:

```powershell
py -m pip install -r requirements.txt
py -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```
