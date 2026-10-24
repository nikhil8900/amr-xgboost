from pathlib import Path
from datetime import timedelta
import random

import pandas as pd
import simpy


# The project folder is the parent of src.
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Try the likely names of the forecast file.
forecast_candidates = [
    OUTPUT_DIR / "next_demand_forecast_xgboost.csv",
    OUTPUT_DIR / "next_demand_forecast.csv",
]

forecast_path = next(
    (path for path in forecast_candidates if path.exists()),
    None,
)

if forecast_path is None:
    names = "\n".join(f"  - {path.name}" for path in forecast_candidates)
    raise FileNotFoundError(
        "Could not find your forecast CSV in the outputs folder.\n"
        f"Expected one of:\n{names}\n"
        "Run your XGBoost prediction script first."
    )

forecast = pd.read_csv(forecast_path)

# Find the forecast value column.
prediction_column = next(
    (
        column
        for column in [
            "predicted_demand_next_15min",
            "xgboost_prediction",
            "prediction",
        ]
        if column in forecast.columns
    ),
    None,
)

if prediction_column is None:
    raise ValueError(
        "Could not find a forecast column. The CSV needs "
        "'predicted_demand_next_15min' or 'xgboost_prediction'."
    )

if "station_id" not in forecast.columns:
    raise ValueError("The forecast CSV must contain a 'station_id' column.")

# Use the latest forecast timestamp if the file contains more than one.
time_column = next(
    (
        column
        for column in ["input_timestamp", "timestamp"]
        if column in forecast.columns
    ),
    None,
)

if time_column:
    forecast[time_column] = pd.to_datetime(forecast[time_column], errors="coerce")
    latest_time = forecast[time_column].max()
    forecast = forecast[forecast[time_column] == latest_time].copy()
else:
    latest_time = pd.Timestamp.now().floor("s")

forecast[prediction_column] = pd.to_numeric(
    forecast[prediction_column], errors="coerce"
).fillna(0).clip(lower=0)

if "zone" not in forecast.columns:
    forecast["zone"] = ""

# Use a fixed seed so the same forecast produces repeatable results.
random.seed(42)

env = simpy.Environment()
simulated_arrivals = []


def generate_station_arrivals(station_id, zone, expected_demand):
    """Simulate request arrival times for one station over 15 minutes."""
    if expected_demand <= 0:
        return

    # Treat the forecast as the expected number of requests in 15 minutes.
    arrival_rate_per_minute = expected_demand / 15.0

    while True:
        minutes_until_arrival = random.expovariate(arrival_rate_per_minute)
        yield env.timeout(minutes_until_arrival)

        if env.now >= 15:
            break

        simulated_arrivals.append(
            {
                "station_id": station_id,
                "zone": zone,
                "arrival_minute": round(env.now, 2),
                "simulated_arrival_time": (
                    latest_time + timedelta(minutes=env.now)
                ).strftime("%Y-%m-%d %H:%M:%S"),
            }
        )


for row in forecast.itertuples(index=False):
    values = row._asdict()
    station_id = values["station_id"]
    zone = values.get("zone", "")
    expected_demand = float(values[prediction_column])

    env.process(
        generate_station_arrivals(station_id, zone, expected_demand)
    )

env.run(until=15)

# Save every simulated arrival.
arrivals_path = OUTPUT_DIR / "simpy_simulated_arrivals.csv"
arrivals = pd.DataFrame(
    simulated_arrivals,
    columns=[
        "station_id",
        "zone",
        "arrival_minute",
        "simulated_arrival_time",
    ],
)
arrivals.to_csv(arrivals_path, index=False)

# Save a per-station summary.
summary = forecast[["station_id", "zone", prediction_column]].copy()
summary = summary.rename(
    columns={prediction_column: "forecast_expected_demand"}
)

if arrivals.empty:
    simulated_counts = pd.Series(dtype="int64")
else:
    simulated_counts = arrivals["station_id"].value_counts()

summary["simulated_requests"] = (
    summary["station_id"].map(simulated_counts).fillna(0).astype(int)
)

summary_path = OUTPUT_DIR / "simpy_simulation_summary.csv"
summary.to_csv(summary_path, index=False)

print(f"Forecast used: {forecast_path.name}")
print(f"Forecast time: {latest_time}")
print(f"Stations simulated: {len(summary)}")
print(f"Total simulated requests: {len(arrivals)}")
print("\nStations with the most simulated requests:")
print(
    summary.sort_values("simulated_requests", ascending=False)
    .head(10)
    .to_string(index=False)
)
print(f"\nSaved arrival details to: {arrivals_path}")
print(f"Saved station summary to: {summary_path}")