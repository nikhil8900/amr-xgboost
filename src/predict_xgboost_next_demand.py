from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = (
    ROOT
    / "data"
    / "AMR_FINAL_30DAY_RESEARCH_READY"
    / "station_state_5min.csv"
)
OUTPUT_DIR = ROOT / "outputs"
MODEL_FILE = OUTPUT_DIR / "xgboost_pipeline.joblib"
STGT_FILE = OUTPUT_DIR / "next_demand_forecast.csv"

TARGET = "demand_next_15min"

CATEGORICAL_FEATURES = ["station_id", "zone"]

BASE_NUMERIC_FEATURES = [
    "x_m",
    "y_m",
    "new_requests",
    "current_requests",
    "recent_demand_15min",
    "queue_length",
    "completed_requests",
    "global_active_amrs",
    "global_idle_amrs",
    "global_charging_amrs",
    "global_maintenance_amrs",
    "global_available_amrs",
    "avg_travel_time_min",
    "avg_travel_time_min_ml",
    "hour",
    "day_of_week",
]

HISTORY_SOURCES = [
    "new_requests",
    "recent_demand_15min",
    "queue_length",
]

TARGET_HISTORY_LAGS = (3, 6, 12)
TARGET_HISTORY_WINDOWS = (3, 12)


def add_history_features(data):
    """Create the same history and time features used during training."""
    result = data.copy()
    result["station_id"] = result["station_id"].astype(str)
    result = result.sort_values(
        ["station_id", "timestamp"]
    ).reset_index(drop=True)

    for column in HISTORY_SOURCES:
        if column not in result.columns:
            result[column] = np.nan

        result[column] = pd.to_numeric(result[column], errors="coerce")
        station_values = result.groupby("station_id", sort=False)[column]

        for lag_steps in (1, 3, 6, 12):
            result[f"{column}_lag_{lag_steps * 5}m"] = station_values.shift(
                lag_steps
            )

        for window_steps in (3, 12):
            result[f"{column}_previous_mean_{window_steps * 5}m"] = (
                result.groupby("station_id", sort=False)[column]
                .transform(
                    lambda values: values.shift(1).rolling(
                        window=window_steps,
                        min_periods=1,
                    ).mean()
                )
            )

    result["hour"] = result["timestamp"].dt.hour
    result["day_of_week"] = result["timestamp"].dt.dayofweek

    minute_of_day = (
        result["timestamp"].dt.hour * 60
        + result["timestamp"].dt.minute
    )
    result["time_of_day_sin"] = np.sin(
        2 * np.pi * minute_of_day / 1440
    )
    result["time_of_day_cos"] = np.cos(
        2 * np.pi * minute_of_day / 1440
    )
    result["day_of_week_sin"] = np.sin(
        2 * np.pi * result["day_of_week"] / 7
    )
    result["day_of_week_cos"] = np.cos(
        2 * np.pi * result["day_of_week"] / 7
    )

    # The latest target value is not used directly. Only completed target
    # windows from at least 15 minutes earlier are included.
    if TARGET not in result.columns:
        result[TARGET] = np.nan
    result[TARGET] = pd.to_numeric(result[TARGET], errors="coerce")

    for lag_steps in TARGET_HISTORY_LAGS:
        result[f"{TARGET}_lag_{lag_steps * 5}m"] = (
            result.groupby("station_id", sort=False)[TARGET]
            .shift(lag_steps)
        )

    for window_steps in TARGET_HISTORY_WINDOWS:
        result[f"{TARGET}_previous_mean_{window_steps * 5}m"] = (
            result.groupby("station_id", sort=False)[TARGET]
            .transform(
                lambda values: values.shift(3).rolling(
                    window=window_steps,
                    min_periods=1,
                ).mean()
            )
        )

    result["zone"] = result["zone"].fillna("Unknown").astype(str)

    return result.sort_values(
        ["timestamp", "station_id"]
    ).reset_index(drop=True)


def main():
    for file_path in (DATA_FILE, MODEL_FILE):
        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}\n"
                "Run py src/train_xgboost.py first."
            )

    artifact = joblib.load(MODEL_FILE)

    if isinstance(artifact, dict):
        model = artifact["model"]
        features = artifact["features"]
        selected_model = artifact.get("selected_model", "saved pipeline")
    else:
        model = artifact
        features = CATEGORICAL_FEATURES + BASE_NUMERIC_FEATURES
        selected_model = "saved pipeline"

    data = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
    data["station_id"] = data["station_id"].astype(str)
    data = add_history_features(data)

    latest_timestamp = data["timestamp"].max()
    latest = data[data["timestamp"] == latest_timestamp].copy()

    for column in features:
        if column not in latest.columns:
            latest[column] = np.nan

    for column in features:
        if column not in CATEGORICAL_FEATURES:
            latest[column] = pd.to_numeric(
                latest[column],
                errors="coerce",
            )

    predictions = np.maximum(model.predict(latest[features]), 0)

    results = latest[["timestamp", "station_id", "zone"]].copy()
    results = results.rename(columns={"timestamp": "input_timestamp"})
    results["predicted_demand_next_15min"] = predictions
    results["xgboost_rank"] = (
        results["predicted_demand_next_15min"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    # Add STGT values for comparison when they match this forecast time.
    if STGT_FILE.exists():
        stgt = pd.read_csv(STGT_FILE, parse_dates=["input_timestamp"])
        stgt["station_id"] = stgt["station_id"].astype(str)
        stgt = stgt[stgt["input_timestamp"] == latest_timestamp]

        if not stgt.empty:
            comparison = stgt[
                ["station_id", "predicted_demand_next_15min"]
            ].rename(
                columns={
                    "predicted_demand_next_15min": "stgt_prediction"
                }
            )

            results = results.merge(
                comparison,
                on="station_id",
                how="left",
            )
            results["stgt_rank"] = (
                results["stgt_prediction"]
                .rank(ascending=False, method="min")
                .astype("Int64")
            )

    results = results.sort_values(
        "predicted_demand_next_15min",
        ascending=False,
    ).reset_index(drop=True)

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_file = OUTPUT_DIR / "next_demand_forecast_xgboost.csv"
    results.to_csv(output_file, index=False)

    print(f"Model selected: {selected_model}")
    print(f"Data used through: {latest_timestamp}")
    print("Forecast target: demand_next_15min")
    print("\nHighest predicted-demand stations:")
    print(results.head(10).to_string(index=False))
    print(f"\nSaved all forecasts to: {output_file}")


if __name__ == "__main__":
    main()