from pathlib import Path

import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

ROOT = Path(__file__).parent.parent
DATA_FILE = (
    ROOT / "data" / "AMR_FINAL_30DAY_RESEARCH_READY"
    / "station_state_5min.csv"
)
COMPARISON_FILE = ROOT / "outputs" / "model_comparison_predictions.csv"
OUTPUT_DIR = ROOT / "outputs"

target = "demand_next_15min"

# Read the original data and the XGBoost predictions.
df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
xgb_results = pd.read_csv(COMPARISON_FILE, parse_dates=["timestamp"])

# Use a complete operating day from the held-out test period.
test_dates = xgb_results["timestamp"].dt.date
counts = xgb_results.assign(test_date=test_dates).groupby("test_date")[
    "timestamp"
].nunique()

full_days = counts[counts >= 100].index
if len(full_days) == 0:
    raise ValueError("Could not find a full operating day in the test period.")

chosen_day = full_days[0]
day_rows = xgb_results[test_dates == chosen_day].copy()

# Sample once per hour. At each sampled time, use target labels that would
# already have finished and become observable 15 minutes earlier.
origins = sorted(
    timestamp
    for timestamp in day_rows["timestamp"].drop_duplicates()
    if timestamp.minute == 0 and timestamp.second == 0
)

if not origins:
    raise ValueError("No hourly prediction times found for the selected day.")

print(f"Checking day: {chosen_day}")
print(f"Hourly prediction times: {len(origins)}")
print("Prophet will refit at each sampled prediction time.")

results = []

for origin in origins:
    latest_known_label_time = origin - pd.Timedelta(minutes=15)

    for station_id in sorted(day_rows["station_id"].unique()):
        history = df[
            (df["station_id"] == station_id)
            & (df["timestamp"] <= latest_known_label_time)
        ][["timestamp", target]].dropna()

        if len(history) < 50:
            continue

        prophet_train = history.rename(
            columns={"timestamp": "ds", target: "y"}
        )

        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            uncertainty_samples=0,
        )
        model.fit(prophet_train)

        forecast = model.predict(pd.DataFrame({"ds": [origin]}))
        prophet_value = max(0.0, float(forecast["yhat"].iloc[0]))

        actual_row = day_rows[
            (day_rows["timestamp"] == origin)
            & (day_rows["station_id"] == station_id)
        ]

        if actual_row.empty:
            continue

        row = actual_row.iloc[0]
        results.append(
            {
                "timestamp": origin,
                "station_id": station_id,
                "actual_demand": row[target],
                "prophet_prediction": prophet_value,
                "xgboost_prediction": row["xgboost_prediction"],
                "recent_demand_baseline": row["recent_demand_15min"],
            }
        )

    print(f"Finished prediction time: {origin}")

result_df = pd.DataFrame(results)
if result_df.empty:
    raise ValueError("No forecasts were produced.")

print(f"\nCompared rows: {len(result_df)}")
for name, column in [
    ("Prophet", "prophet_prediction"),
    ("XGBoost", "xgboost_prediction"),
    ("Baseline", "recent_demand_baseline"),
]:
    actual = result_df["actual_demand"]
    prediction = pd.to_numeric(result_df[column], errors="coerce").fillna(0)

    mae = mean_absolute_error(actual, prediction)
    rmse = root_mean_squared_error(actual, prediction)
    print(f"{name:8s} MAE={mae:.3f}, RMSE={rmse:.3f}")

output_file = OUTPUT_DIR / "rolling_prophet_check.csv"
result_df.to_csv(output_file, index=False)
print(f"\nSaved hourly rolling comparison to: {output_file}")