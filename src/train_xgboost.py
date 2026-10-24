from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = (
    ROOT
    / "data"
    / "AMR_FINAL_30DAY_RESEARCH_READY"
    / "station_state_5min.csv"
)
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "demand_next_15min"
HORIZON_MINUTES = 15

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

TIME_CYCLE_FEATURES = [
    "time_of_day_sin",
    "time_of_day_cos",
    "day_of_week_sin",
    "day_of_week_cos",
]

# Each row is five minutes. A three-row target lag is 15 minutes.
# Shorter target lags could include demand that had not arrived yet.
TARGET_HISTORY_LAGS = (3, 6, 12)
TARGET_HISTORY_WINDOWS = (3, 12)

TARGET_HISTORY_FEATURES = [
    *[
        f"{TARGET}_lag_{steps * 5}m"
        for steps in TARGET_HISTORY_LAGS
    ],
    *[
        f"{TARGET}_previous_mean_{steps * 5}m"
        for steps in TARGET_HISTORY_WINDOWS
    ],
]


def add_history_features(data):
    """Create features using current and earlier observations only."""
    result = data.copy()
    result["station_id"] = result["station_id"].astype(str)
    result = result.sort_values(
        ["station_id", "timestamp"]
    ).reset_index(drop=True)

    for column in HISTORY_SOURCES:
        result[column] = pd.to_numeric(result[column], errors="coerce")

        station_values = result.groupby("station_id", sort=False)[column]

        # Earlier observed station values. Each row is five minutes.
        for lag_steps in (1, 3, 6, 12):
            lag_minutes = lag_steps * 5
            result[f"{column}_lag_{lag_minutes}m"] = station_values.shift(
                lag_steps
            )

        # shift(1) excludes the current row from these averages.
        for window_steps in (3, 12):
            window_minutes = window_steps * 5
            result[f"{column}_previous_mean_{window_minutes}m"] = (
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

    # Cyclical time features help represent that 23:00 is near 00:00,
    # and Sunday is near Monday.
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

    # Use only completed past target windows. The 15-minute lag ends
    # at the current timestamp, so it does not include future requests.
    result[TARGET] = pd.to_numeric(result[TARGET], errors="coerce")
    target_values = result.groupby("station_id", sort=False)[TARGET]

    for lag_steps in TARGET_HISTORY_LAGS:
        result[f"{TARGET}_lag_{lag_steps * 5}m"] = target_values.shift(
            lag_steps
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


df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])

required_columns = [
    "timestamp",
    "station_id",
    "zone",
    TARGET,
    *BASE_NUMERIC_FEATURES[:-2],
]
missing = [column for column in required_columns if column not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in dataset: {missing}")

df["station_id"] = df["station_id"].astype(str)
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.dropna(subset=["timestamp", TARGET]).copy()

if (df[TARGET] < 0).any():
    raise ValueError(f"{TARGET} contains negative demand values.")

# Create historical features before splitting. Each uses earlier rows
# from the same station.
df = add_history_features(df)

timestamps = sorted(df["timestamp"].drop_duplicates().tolist())
if len(timestamps) < 20:
    raise ValueError(
        "Not enough timestamps for chronological train/validation/test splits."
    )

validation_start = timestamps[int(len(timestamps) * 0.70)]
test_start = timestamps[int(len(timestamps) * 0.85)]

# Remove training labels whose 15-minute target windows would overlap
# the next split.
validation_train_cutoff = validation_start - pd.Timedelta(
    minutes=HORIZON_MINUTES
)
final_train_cutoff = test_start - pd.Timedelta(
    minutes=HORIZON_MINUTES
)

validation_train = df[df["timestamp"] < validation_train_cutoff].copy()
validation = df[
    (df["timestamp"] >= validation_start)
    & (df["timestamp"] < test_start)
].copy()
final_train = df[df["timestamp"] < final_train_cutoff].copy()
test = df[df["timestamp"] >= test_start].copy()

if any(part.empty for part in (validation_train, validation, final_train, test)):
    raise ValueError("A data split is empty; check the dataset length.")

print(f"Validation training cutoff: {validation_train_cutoff}")
print(f"Validation starts:          {validation_start}")
print(f"Final training cutoff:      {final_train_cutoff}")
print(f"Test starts:                {test_start}")
print(f"Validation training rows:   {len(validation_train):,}")
print(f"Validation rows:            {len(validation):,}")
print(f"Final training rows:        {len(final_train):,}")
print(f"Test rows:                  {len(test):,}")

HISTORY_FEATURES = []
for column in HISTORY_SOURCES:
    for lag_steps in (1, 3, 6, 12):
        HISTORY_FEATURES.append(f"{column}_lag_{lag_steps * 5}m")
    for window_steps in (3, 12):
        HISTORY_FEATURES.append(
            f"{column}_previous_mean_{window_steps * 5}m"
        )

# Compare the original features with several experimental feature sets.
FEATURE_SETS = {
    "existing_features": BASE_NUMERIC_FEATURES,
    "with_existing_history": BASE_NUMERIC_FEATURES + HISTORY_FEATURES,
    "with_time_cycles": BASE_NUMERIC_FEATURES + TIME_CYCLE_FEATURES,
    "with_safe_target_history": (
        BASE_NUMERIC_FEATURES + TARGET_HISTORY_FEATURES
    ),
    "with_all_experimental_features": (
        BASE_NUMERIC_FEATURES
        + HISTORY_FEATURES
        + TIME_CYCLE_FEATURES
        + TARGET_HISTORY_FEATURES
    ),
}

MODEL_OPTIONS = [
    {
        "name": "square_depth3",
        "objective": "reg:squarederror",
        "n_estimators": 500,
        "max_depth": 3,
        "learning_rate": 0.03,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 5.0,
    },
    {
        "name": "square_depth4",
        "objective": "reg:squarederror",
        "n_estimators": 600,
        "max_depth": 4,
        "learning_rate": 0.03,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 5.0,
    },
    {
        "name": "square_depth6",
        "objective": "reg:squarederror",
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "min_child_weight": 1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_lambda": 1.0,
    },
    {
        "name": "poisson_depth3",
        "objective": "count:poisson",
        "n_estimators": 500,
        "max_depth": 3,
        "learning_rate": 0.03,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 5.0,
    },
]


def make_pipeline(numeric_features, options):
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categories",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("numbers", "passthrough", numeric_features),
        ]
    )

    model = XGBRegressor(
        objective=options["objective"],
        n_estimators=options["n_estimators"],
        max_depth=options["max_depth"],
        learning_rate=options["learning_rate"],
        min_child_weight=options["min_child_weight"],
        subsample=options["subsample"],
        colsample_bytree=options["colsample_bytree"],
        reg_lambda=options["reg_lambda"],
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("xgboost", model),
        ]
    )


validation_scores = []
best_model = None
best_numeric_features = None
best_features = None
best_description = None
best_validation_mae = float("inf")

print("\nSelecting features and model settings using validation data:")

for feature_set_name, numeric_features in FEATURE_SETS.items():
    features = CATEGORICAL_FEATURES + numeric_features

    for options in MODEL_OPTIONS:
        description = f"{feature_set_name} / {options['name']}"
        model = make_pipeline(numeric_features, options)
        model.fit(validation_train[features], validation_train[TARGET])

        predictions = np.maximum(model.predict(validation[features]), 0)
        mae = mean_absolute_error(validation[TARGET], predictions)
        rmse = root_mean_squared_error(validation[TARGET], predictions)

        validation_scores.append(
            {
                "model": description,
                "validation_mae": mae,
                "validation_rmse": rmse,
            }
        )
        print(f"  {description:52s} MAE={mae:.3f}, RMSE={rmse:.3f}")

        if mae < best_validation_mae:
            best_validation_mae = mae
            best_model = options
            best_numeric_features = numeric_features
            best_features = features
            best_description = description

pd.DataFrame(validation_scores).to_csv(
    OUTPUT_DIR / "xgboost_validation_model_comparison.csv",
    index=False,
)

print(f"\nSelected on validation data: {best_description}")

# Train the selected model using all data available before the test period.
xgb_model = make_pipeline(best_numeric_features, best_model)
xgb_model.fit(final_train[best_features], final_train[TARGET])

xgb_predictions = np.maximum(
    xgb_model.predict(test[best_features]),
    0,
)

# Prophet comparison on the same final training and test periods.
prophet_predictions = []

for station_id in sorted(test["station_id"].unique()):
    station_train = final_train[final_train["station_id"] == station_id]
    station_test = test[test["station_id"] == station_id]

    if station_train.empty or station_test.empty:
        continue

    prophet_train = station_train[["timestamp", TARGET]].rename(
        columns={"timestamp": "ds", TARGET: "y"}
    )

    prophet_model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        interval_width=0.80,
    )
    prophet_model.fit(prophet_train)

    future = station_test[["timestamp"]].rename(
        columns={"timestamp": "ds"}
    )
    prophet_forecast = prophet_model.predict(future)

    station_result = station_test[
        ["timestamp", "station_id", TARGET]
    ].copy()
    station_result["prophet_prediction"] = (
        prophet_forecast["yhat"].clip(lower=0).to_numpy()
    )
    prophet_predictions.append(station_result)

results = test[
    ["timestamp", "station_id", TARGET, "recent_demand_15min"]
].copy()
results["xgboost_prediction"] = xgb_predictions

if prophet_predictions:
    prophet_results = pd.concat(prophet_predictions, ignore_index=True)
    results = results.merge(
        prophet_results[
            ["timestamp", "station_id", "prophet_prediction"]
        ],
        on=["timestamp", "station_id"],
        how="left",
    )
else:
    results["prophet_prediction"] = np.nan

baseline = pd.to_numeric(
    results["recent_demand_15min"],
    errors="coerce",
).fillna(0).clip(lower=0)

print("\nFinal test results:")
for name, predictions in [
    ("XGBoost", results["xgboost_prediction"]),
    ("Prophet", results["prophet_prediction"]),
    ("Baseline", baseline),
]:
    valid = predictions.notna()
    mae = mean_absolute_error(
        results.loc[valid, TARGET],
        predictions.loc[valid],
    )
    rmse = root_mean_squared_error(
        results.loc[valid, TARGET],
        predictions.loc[valid],
    )
    print(f"  {name:8s} MAE={mae:.3f}, RMSE={rmse:.3f}")

results.to_csv(OUTPUT_DIR / "model_comparison_predictions.csv", index=False)

# Save the selected model and its feature list for prediction time.
joblib.dump(
    {
        "model": xgb_model,
        "features": best_features,
        "selected_model": best_description,
    },
    OUTPUT_DIR / "xgboost_pipeline.joblib",
)

xgb_model.named_steps["xgboost"].save_model(
    OUTPUT_DIR / "xgboost_model.json"
)

print("\nSaved validation comparison to:")
print(f"  {OUTPUT_DIR / 'xgboost_validation_model_comparison.csv'}")
print("Saved test predictions and model in:")
print(f"  {OUTPUT_DIR}")