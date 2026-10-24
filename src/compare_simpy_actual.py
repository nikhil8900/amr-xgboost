from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "AMR_FINAL_30DAY_RESEARCH_READY"
OUTPUT_DIR = ROOT / "outputs"

FORECAST_FILE = OUTPUT_DIR / "next_demand_forecast_xgboost.csv"
REQUESTS_FILE = DATA_DIR / "transport_requests.csv"
SIMPY_SUMMARY_FILE = OUTPUT_DIR / "simpy_simulation_summary.csv"
SIMPY_ARRIVALS_FILE = OUTPUT_DIR / "simpy_simulated_arrivals.csv"
RESULT_FILE = OUTPUT_DIR / "actual_vs_forecast_vs_simpy.csv"


def main():
    required_files = [
        FORECAST_FILE,
        REQUESTS_FILE,
        SIMPY_SUMMARY_FILE,
        SIMPY_ARRIVALS_FILE,
    ]

    missing_files = [path for path in required_files if not path.exists()]
    if missing_files:
        missing_text = "\n".join(str(path) for path in missing_files)
        raise FileNotFoundError(f"Missing required file(s):\n{missing_text}")

    forecast = pd.read_csv(
        FORECAST_FILE,
        parse_dates=["input_timestamp"],
    )
    requests = pd.read_csv(
        REQUESTS_FILE,
        parse_dates=["request_timestamp"],
    )
    simpy_summary = pd.read_csv(SIMPY_SUMMARY_FILE)
    simpy_arrivals = pd.read_csv(SIMPY_ARRIVALS_FILE)

    forecast["station_id"] = forecast["station_id"].astype(str)
    requests["source_station"] = requests["source_station"].astype(str)
    simpy_summary["station_id"] = simpy_summary["station_id"].astype(str)
    simpy_arrivals["station_id"] = simpy_arrivals["station_id"].astype(str)

    # Use the latest forecast timestamp in the forecast file.
    forecast_time = forecast["input_timestamp"].max()
    forecast = forecast[forecast["input_timestamp"] == forecast_time].copy()

    if forecast.empty:
        raise ValueError("The forecast file contains no usable forecast rows.")

    end_time = forecast_time + pd.Timedelta(minutes=15)

    # Compare against requests that actually arrived in the next 15 minutes.
    actual_window = requests[
        (requests["request_timestamp"] > forecast_time)
        & (requests["request_timestamp"] <= end_time)
    ]

    actual_counts = (
        actual_window.groupby("source_station")
        .size()
        .rename("actual_requests")
        .reset_index()
        .rename(columns={"source_station": "station_id"})
    )

    # Confirm the SimPy summary was generated from this forecast.
    forecast_check = forecast[
        ["station_id", "predicted_demand_next_15min"]
    ].merge(
        simpy_summary[["station_id", "forecast_expected_demand"]],
        on="station_id",
        how="left",
        validate="one_to_one",
    )

    if forecast_check["forecast_expected_demand"].isna().any():
        raise ValueError(
            "Some forecast stations are missing from the SimPy summary. "
            "Rerun the prediction and simulation scripts."
        )

    mismatch = (
        forecast_check["predicted_demand_next_15min"]
        - forecast_check["forecast_expected_demand"]
    ).abs()

    if (mismatch > 0.02).any():
        raise ValueError(
            "The SimPy summary does not match the current forecast. "
            "Rerun these first:\n"
            "py src/predict_xgboost_next_demand.py\n"
            "py src/simulate_forecast_demand.py"
        )

    # Check that the summary counts agree with the saved SimPy arrivals.
    arrival_counts = (
        simpy_arrivals.groupby("station_id")
        .size()
        .rename("arrival_file_count")
        .reset_index()
    )

    simpy_summary = simpy_summary.merge(
        arrival_counts,
        on="station_id",
        how="left",
    )
    simpy_summary["arrival_file_count"] = (
        simpy_summary["arrival_file_count"].fillna(0).astype(int)
    )
    simpy_summary["simulated_requests"] = pd.to_numeric(
        simpy_summary["simulated_requests"],
        errors="coerce",
    ).fillna(0)

    if (
        simpy_summary["simulated_requests"].round().astype(int)
        != simpy_summary["arrival_file_count"]
    ).any():
        raise ValueError(
            "The SimPy summary counts do not match the arrivals file. "
            "Rerun py src/simulate_forecast_demand.py."
        )

    # Build one row per station for the comparison.
    results = forecast[
        [
            "station_id",
            "zone",
            "predicted_demand_next_15min",
        ]
    ].merge(
        simpy_summary[
            ["station_id", "simulated_requests"]
        ],
        on="station_id",
        how="left",
        validate="one_to_one",
    )

    results = results.merge(
        actual_counts,
        on="station_id",
        how="left",
        validate="one_to_one",
    )

    results["actual_requests"] = results["actual_requests"].fillna(0).astype(int)
    results["simulated_requests"] = (
        results["simulated_requests"].fillna(0).round().astype(int)
    )

    results["forecast_absolute_error"] = (
        results["predicted_demand_next_15min"] - results["actual_requests"]
    ).abs()

    results["simpy_absolute_error"] = (
        results["simulated_requests"] - results["actual_requests"]
    ).abs()

    results = results.sort_values(
        "predicted_demand_next_15min",
        ascending=False,
    ).reset_index(drop=True)

    OUTPUT_DIR.mkdir(exist_ok=True)
    results.to_csv(RESULT_FILE, index=False)

    forecast_mae = results["forecast_absolute_error"].mean()
    simpy_mae = results["simpy_absolute_error"].mean()

    print(f"Forecast time: {forecast_time}")
    print(f"Actual request window: after {forecast_time} through {end_time}")
    print(f"Stations compared: {len(results)}")
    print()
    print(f"XGBoost forecast total: {results['predicted_demand_next_15min'].sum():.1f}")
    print(f"SimPy simulated total:  {results['simulated_requests'].sum()}")
    print(f"Actual request total:   {results['actual_requests'].sum()}")
    print()
    print(f"XGBoost per-station MAE vs actual: {forecast_mae:.3f}")
    print(f"SimPy per-station MAE vs actual:   {simpy_mae:.3f}")
    print()
    print("Per-station comparison:")
    print(
        results[
            [
                "station_id",
                "predicted_demand_next_15min",
                "simulated_requests",
                "actual_requests",
                "forecast_absolute_error",
                "simpy_absolute_error",
            ]
        ].to_string(index=False, formatters={
            "predicted_demand_next_15min": "{:.2f}".format,
            "forecast_absolute_error": "{:.2f}".format,
            "simpy_absolute_error": "{:.2f}".format,
        })
    )
    print(f"\nSaved comparison to: {RESULT_FILE}")
    print(
        "\nNote: XGBoost forecast vs actual requests measures forecast accuracy. "
        "SimPy generates a random scenario using the forecast, so its difference "
        "from actual requests is illustrative."
    )


if __name__ == "__main__":
    main()