from pathlib import Path

import numpy as np
import pandas as pd


input_file = Path("outputs/stgt_predictions.csv")

if not input_file.exists():
    raise FileNotFoundError(f"Could not find {input_file}")

data = pd.read_csv(input_file)

prediction_columns = [
    "stgt_prediction",
    "xgboost_prediction",
    "recent_demand_baseline",
]

missing = [column for column in prediction_columns if column not in data.columns]
if missing:
    raise ValueError(f"Missing columns in the CSV: {missing}")

# Keep only rows where all three models have predictions, so the
# comparison uses the same test rows for each model.
common = data.dropna(
    subset=prediction_columns + ["actual_demand"]
).copy()

if common.empty:
    raise ValueError("No shared prediction rows were found.")

common["station_id"] = common["station_id"].astype(str)

print("Shared test timestamps:", common["timestamp"].nunique())
print()

for column, model_name in [
    ("stgt_prediction", "STGT"),
    ("xgboost_prediction", "XGBoost"),
    ("recent_demand_baseline", "Baseline"),
]:
    overlaps = []

    for _, group in common.groupby("timestamp"):
        predicted_top_5 = set(
            group.nlargest(5, column)["station_id"]
        )
        actual_top_5 = set(
            group.nlargest(5, "actual_demand")["station_id"]
        )

        overlaps.append(len(predicted_top_5 & actual_top_5) / 5)

    mae = (
        common[column] - common["actual_demand"]
    ).abs().mean()

    average_overlap_percent = 100 * np.mean(overlaps)

    print(
        f"{model_name}: "
        f"MAE={mae:.3f}, "
        f"average top-5 overlap={average_overlap_percent:.1f}%"
    )