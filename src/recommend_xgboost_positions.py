from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "AMR_FINAL_30DAY_RESEARCH_READY"
OUTPUT_DIR = ROOT / "outputs"

AMR_FILE = DATA_DIR / "amr_state_5min.csv"
STATION_FILE = DATA_DIR / "station_master.csv"
FORECAST_FILE = OUTPUT_DIR / "next_demand_forecast_xgboost.csv"

MIN_BATTERY_PCT = 20.0
NUMBER_OF_STATIONS = 5


def main():
    for file_path in [AMR_FILE, STATION_FILE, FORECAST_FILE]:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

    amrs = pd.read_csv(AMR_FILE)
    stations = pd.read_csv(STATION_FILE)
    forecasts = pd.read_csv(FORECAST_FILE)

    amrs["timestamp"] = pd.to_datetime(
        amrs["timestamp"], errors="coerce"
    )
    amrs["available_at_timestamp"] = pd.to_datetime(
        amrs["available_at_timestamp"], errors="coerce"
    )
    forecasts["input_timestamp"] = pd.to_datetime(
        forecasts["input_timestamp"], errors="coerce"
    )

    forecast_time = forecasts["input_timestamp"].max()
    latest_amr_time = amrs["timestamp"].max()

    if forecast_time != latest_amr_time:
        raise ValueError(
            "The XGBoost forecast and latest AMR snapshot are for "
            "different timestamps.\n"
            f"Forecast: {forecast_time}\n"
            f"AMR snapshot: {latest_amr_time}"
        )

    snapshot = amrs[amrs["timestamp"] == forecast_time].copy()

    for column in ["battery_pct", "x_m", "y_m"]:
        snapshot[column] = pd.to_numeric(
            snapshot[column], errors="coerce"
        )

    eligible = snapshot[
        snapshot["state"].astype(str).str.lower().eq("idle")
        & (snapshot["battery_pct"] >= MIN_BATTERY_PCT)
        & (snapshot["available_at_timestamp"] <= forecast_time)
    ].dropna(subset=["x_m", "y_m"]).copy()

    eligible_count = len(eligible)

    if eligible.empty:
        raise ValueError(
            "No idle AMRs meet the battery and availability settings."
        )

    stations["station_id"] = stations["station_id"].astype(str)
    forecasts["station_id"] = forecasts["station_id"].astype(str)

    forecast_stations = forecasts.merge(
        stations[["station_id", "x_m", "y_m"]],
        on="station_id",
        how="left",
        validate="one_to_one",
    )

    if forecast_stations[["x_m", "y_m"]].isna().any().any():
        raise ValueError(
            "Some forecast stations have no coordinates in station_master.csv."
        )

    priority_stations = forecast_stations.sort_values(
        "predicted_demand_next_15min",
        ascending=False,
    ).head(NUMBER_OF_STATIONS)

    recommendations = []

    # Process the highest predicted-demand station first.
    # Each AMR can be assigned only once.
    for station in priority_stations.itertuples(index=False):
        if eligible.empty:
            break

        distances = np.sqrt(
            (eligible["x_m"] - station.x_m) ** 2
            + (eligible["y_m"] - station.y_m) ** 2
        )

        amr_index = distances.idxmin()
        amr = eligible.loc[amr_index]

        recommendations.append(
            {
                "input_timestamp": forecast_time,
                "station_id": station.station_id,
                "zone": station.zone,
                "predicted_demand_next_15min": (
                    station.predicted_demand_next_15min
                ),
                "recommended_amr_id": amr["amr_id"],
                "amr_battery_pct": amr["battery_pct"],
                "amr_x_m": amr["x_m"],
                "amr_y_m": amr["y_m"],
                "station_x_m": station.x_m,
                "station_y_m": station.y_m,
                "straight_line_distance_m": distances.loc[amr_index],
            }
        )

        eligible = eligible.drop(index=amr_index)

    result = pd.DataFrame(recommendations)

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_file = OUTPUT_DIR / "xgboost_allocation_recommendations.csv"
    result.to_csv(output_file, index=False)

    idle_count = snapshot[
        snapshot["state"].astype(str).str.lower().eq("idle")
    ].shape[0]

    print(f"Forecast and AMR snapshot: {forecast_time}")
    print(f"Idle AMRs: {idle_count}")
    print(
        f"Eligible idle AMRs with battery at least "
        f"{MIN_BATTERY_PCT:.0f}%: {eligible_count}"
    )
    print(f"Recommendations created: {len(result)}")
    print("\nSuggested XGBoost-based pre-positioning:")
    print(result.to_string(index=False))
    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()