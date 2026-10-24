from pathlib import Path

import numpy as np
import pandas as pd
import simpy


ROOT = Path(__file__).resolve().parent.parent
FORECAST_FILE = ROOT / "outputs" / "next_demand_forecast_xgboost.csv"
REQUESTS_FILE = (
    ROOT
    / "data"
    / "AMR_FINAL_30DAY_RESEARCH_READY"
    / "transport_requests.csv"
)
OUTPUT_DIR = ROOT / "outputs"

HORIZON_MINUTES = 15
RANDOM_SEED = 42
FORECAST_COLUMN = "predicted_demand_next_15min"


def wape(actual, predicted):
    """Weighted absolute percentage error."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    actual_total = np.abs(actual).sum()

    if actual_total == 0:
        return float("nan")

    return float(np.abs(actual - predicted).sum() / actual_total * 100)


def station_arrivals(env, station_id, expected_requests, rng, arrival_rows):
    """Generate arrivals for one station using its forecast as the mean."""
    if expected_requests <= 0:
        return

    rate_per_minute = expected_requests / HORIZON_MINUTES

    while True:
        # Random time between arrivals, based on the station's forecast.
        delay = float(rng.exponential(1.0 / rate_per_minute))
        yield env.timeout(delay)

        if env.now > HORIZON_MINUTES:
            break

        arrival_time = forecast_time + pd.Timedelta(minutes=env.now)

        arrival_rows.append(
            {
                "station_id": station_id,
                "arrival_minute": round(float(env.now), 3),
                "simulated_arrival_time": arrival_time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            }
        )


def main():
    for file_path in [FORECAST_FILE, REQUESTS_FILE]:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

    forecast = pd.read_csv(
        FORECAST_FILE,
        parse_dates=["input_timestamp"],
    )

    required_forecast_columns = [
        "input_timestamp",
        "station_id",
        "zone",
        FORECAST_COLUMN,
    ]
    missing = [
        column
        for column in required_forecast_columns
        if column not in forecast.columns
    ]
    if missing:
        raise ValueError(f"Forecast file is missing columns: {missing}")

    forecast_times = forecast["input_timestamp"].dropna().unique()
    if len(forecast_times) != 1:
        raise ValueError(
            "The forecast CSV must contain one input_timestamp. "
            "Run the prediction script first."
        )

    global forecast_time
    forecast_time = pd.Timestamp(forecast_times[0])
    forecast_end = forecast_time + pd.Timedelta(
        minutes=HORIZON_MINUTES
    )

    forecast["station_id"] = forecast["station_id"].astype(str)
    forecast[FORECAST_COLUMN] = pd.to_numeric(
        forecast[FORECAST_COLUMN],
        errors="coerce",
    ).fillna(0).clip(lower=0)

    if forecast["station_id"].duplicated().any():
        raise ValueError("Forecast file contains duplicate station IDs.")

    requests = pd.read_csv(
        REQUESTS_FILE,
        parse_dates=["request_timestamp"],
    )

    required_request_columns = [
        "request_timestamp",
        "source_station",
    ]
    missing = [
        column
        for column in required_request_columns
        if column not in requests.columns
    ]
    if missing:
        raise ValueError(f"Requests file is missing columns: {missing}")

    latest_actual_time = requests["request_timestamp"].max()
    if pd.isna(latest_actual_time) or latest_actual_time < forecast_end:
        raise ValueError(
            "The actual request data does not cover the full 15-minute "
            f"forecast window. It ends at {latest_actual_time}; "
            f"the window ends at {forecast_end}."
        )

    # Count actual arrivals after the forecast time through the end
    # of the 15-minute window.
    actual_window = requests[
        (requests["request_timestamp"] > forecast_time)
        & (requests["request_timestamp"] <= forecast_end)
    ].copy()

    actual_window["source_station"] = (
        actual_window["source_station"].astype(str)
    )
    actual_counts = actual_window.groupby("source_station").size()

    # Run one SimPy arrival process for each station.
    env = simpy.Environment()
    rng = np.random.default_rng(RANDOM_SEED)
    simulated_arrivals = []

    for row in forecast.itertuples(index=False):
        station_id = str(row.station_id)
        expected_requests = float(
            getattr(row, FORECAST_COLUMN)
        )

        env.process(
            station_arrivals(
                env=env,
                station_id=station_id,
                expected_requests=expected_requests,
                rng=rng,
                arrival_rows=simulated_arrivals,
            )
        )

    # The tiny amount past 15 lets SimPy process an event exactly at
    # the boundary; station_arrivals filters out events after minute 15.
    env.run(until=HORIZON_MINUTES + 1e-8)

    arrivals_df = pd.DataFrame(
        simulated_arrivals,
        columns=[
            "station_id",
            "arrival_minute",
            "simulated_arrival_time",
        ],
    )

    simulated_counts = arrivals_df["station_id"].value_counts()

    comparison = forecast[
        ["station_id", "zone", FORECAST_COLUMN]
    ].copy()

    comparison["actual_requests"] = (
        comparison["station_id"]
        .map(actual_counts)
        .fillna(0)
        .astype(int)
    )
    comparison["simulated_requests"] = (
        comparison["station_id"]
        .map(simulated_counts)
        .fillna(0)
        .astype(int)
    )

    comparison["forecast_absolute_error"] = (
        comparison[FORECAST_COLUMN] - comparison["actual_requests"]
    ).abs()
    comparison["simpy_absolute_error"] = (
        comparison["simulated_requests"] - comparison["actual_requests"]
    ).abs()

    actual = comparison["actual_requests"].to_numpy()
    forecast_values = comparison[FORECAST_COLUMN].to_numpy()
    simulated_values = comparison["simulated_requests"].to_numpy()

    forecast_mae = float(np.abs(forecast_values - actual).mean())
    simpy_mae = float(np.abs(simulated_values - actual).mean())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    arrivals_file = OUTPUT_DIR / "simpy_simulated_arrivals.csv"
    comparison_file = OUTPUT_DIR / "actual_vs_forecast_vs_simpy.csv"
    summary_file = OUTPUT_DIR / "simpy_simulation_summary.csv"

    arrivals_df.to_csv(arrivals_file, index=False)
    comparison.sort_values(
        FORECAST_COLUMN,
        ascending=False,
    ).to_csv(comparison_file, index=False)

    summary = comparison[
        [
            "station_id",
            "zone",
            FORECAST_COLUMN,
            "simulated_requests",
            "actual_requests",
        ]
    ].rename(
        columns={
            FORECAST_COLUMN: "forecast_expected_demand",
        }
    )
    summary.to_csv(summary_file, index=False)

    print(f"Forecast time: {forecast_time}")
    print(f"Actual request window: after {forecast_time} through {forecast_end}")
    print(f"Stations compared: {len(comparison)}")
    print()
    print(f"XGBoost forecast total: {forecast_values.sum():.1f}")
    print(f"SimPy simulated total:  {int(simulated_values.sum())}")
    print(f"Actual arrival total:   {int(actual.sum())}")
    print()
    print(f"XGBoost vs actual: MAE={forecast_mae:.3f}, "
          f"WAPE={wape(actual, forecast_values):.2f}%")
    print(f"SimPy vs actual:   MAE={simpy_mae:.3f}, "
          f"WAPE={wape(actual, simulated_values):.2f}%")
    print()
    print("Per-station results:")
    print(
        comparison.sort_values(
            "actual_requests",
            ascending=False,
        ).to_string(index=False)
    )
    print()
    print(f"Saved SimPy arrivals to: {arrivals_file}")
    print(f"Saved comparison to: {comparison_file}")
    print(f"Saved summary to: {summary_file}")
    print()
    print(
        "The XGBoost forecast vs actual measures forecast accuracy. "
        "SimPy creates a random arrival scenario from that forecast, "
        "so its simulated count can differ from actual arrivals."
    )


if __name__ == "__main__":
    main()