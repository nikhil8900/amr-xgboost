from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "AMR_FINAL_30DAY_RESEARCH_READY"
OUTPUT_DIR = ROOT / "outputs"

AMR_FILE = DATA_DIR / "amr_state_5min.csv"
STATION_FILE = DATA_DIR / "station_master.csv"
REQUEST_FILE = DATA_DIR / "transport_requests.csv"
PREDICTION_FILE = OUTPUT_DIR / "stgt_predictions.csv"

MIN_BATTERY_PCT = 20.0
NUMBER_OF_STATIONS_TO_PREPOSITION = 1
WINDOW_MINUTES = 15

MODEL_COLUMNS = {
    "XGBoost": "xgboost_prediction",
    "STGT": "stgt_prediction",
}


def prepare_eligible_amrs(snapshot, timestamp):
    """Return idle AMRs meeting the battery and availability rules."""
    amrs = snapshot.copy()

    for column in ["battery_pct", "x_m", "y_m"]:
        amrs[column] = pd.to_numeric(amrs[column], errors="coerce")

    eligible = amrs[
        amrs["state"].astype(str).str.lower().eq("idle")
        & (amrs["battery_pct"] >= MIN_BATTERY_PCT)
        & (amrs["available_at_timestamp"] <= timestamp)
    ].dropna(subset=["x_m", "y_m"])

    return eligible.reset_index(drop=True)


def preposition_amrs(eligible, forecast_rows, prediction_column, station_lookup):
    """
    Move up to five selected AMRs to the highest forecast stations.
    Return their updated positions and the straight-line staging distance.
    """
    positions = eligible.copy()
    already_staged = set()
    staging_distance = 0.0

    priority_stations = forecast_rows.sort_values(
        prediction_column,
        ascending=False,
    ).head(NUMBER_OF_STATIONS_TO_PREPOSITION)

    for row in priority_stations.itertuples(index=False):
        station_id = str(row.station_id)
        if station_id not in station_lookup:
            continue

        station_x, station_y = station_lookup[station_id]

        candidates = positions[
            ~positions["amr_id"].astype(str).isin(already_staged)
        ]

        if candidates.empty:
            break

        distances = np.sqrt(
            (candidates["x_m"] - station_x) ** 2
            + (candidates["y_m"] - station_y) ** 2
        )

        selected_index = distances.idxmin()
        selected_amr_id = str(positions.loc[selected_index, "amr_id"])

        staging_distance += float(distances.loc[selected_index])
        positions.loc[selected_index, "x_m"] = station_x
        positions.loc[selected_index, "y_m"] = station_y
        already_staged.add(selected_amr_id)

    return positions.reset_index(drop=True), staging_distance


def assign_requests_to_nearest_amrs(requests, eligible, station_lookup):
    """
    Assign each request to its nearest remaining AMR.
    An AMR is used at most once in this 15-minute window.
    """
    positions = eligible.copy().reset_index(drop=True)
    total_response_distance = 0.0
    assigned_count = 0

    ordered_requests = requests.sort_values("request_timestamp")

    for request in ordered_requests.itertuples(index=False):
        source_station = str(request.source_station)

        if source_station not in station_lookup or positions.empty:
            continue

        source_x, source_y = station_lookup[source_station]

        distances = np.sqrt(
            (positions["x_m"] - source_x) ** 2
            + (positions["y_m"] - source_y) ** 2
        )

        nearest_index = distances.idxmin()
        total_response_distance += float(distances.loc[nearest_index])
        assigned_count += 1

        # Prevent assigning the same AMR to another task in this window.
        positions = positions.drop(index=nearest_index).reset_index(drop=True)

    return assigned_count, total_response_distance


def main():
    required_files = [
        AMR_FILE,
        STATION_FILE,
        REQUEST_FILE,
        PREDICTION_FILE,
    ]
    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

    amrs = pd.read_csv(AMR_FILE)
    stations = pd.read_csv(STATION_FILE)
    requests = pd.read_csv(REQUEST_FILE)
    predictions = pd.read_csv(PREDICTION_FILE)

    amrs["timestamp"] = pd.to_datetime(amrs["timestamp"], errors="coerce")
    amrs["available_at_timestamp"] = pd.to_datetime(
        amrs["available_at_timestamp"], errors="coerce"
    )
    requests["request_timestamp"] = pd.to_datetime(
        requests["request_timestamp"], errors="coerce"
    )
    predictions["timestamp"] = pd.to_datetime(
        predictions["timestamp"], errors="coerce"
    )

    amrs["amr_id"] = amrs["amr_id"].astype(str)
    stations["station_id"] = stations["station_id"].astype(str)
    requests["source_station"] = requests["source_station"].astype(str)
    predictions["station_id"] = predictions["station_id"].astype(str)

    for column in MODEL_COLUMNS.values():
        predictions[column] = pd.to_numeric(
            predictions[column], errors="coerce"
        )

    stations["x_m"] = pd.to_numeric(stations["x_m"], errors="coerce")
    stations["y_m"] = pd.to_numeric(stations["y_m"], errors="coerce")

    station_lookup = {
        str(row.station_id): (float(row.x_m), float(row.y_m))
        for row in stations.itertuples(index=False)
        if pd.notna(row.x_m) and pd.notna(row.y_m)
    }

    # Exclude 20:50 and 20:55 forecasts because their 15-minute targets
    # extend beyond the 21:00 shift end in this dataset.
    padding_rows = (
        (predictions["timestamp"].dt.hour == 20)
        & (predictions["timestamp"].dt.minute > 45)
    )
    predictions = predictions[~padding_rows].copy()

    detail_rows = []

    for timestamp, forecast_rows in predictions.groupby("timestamp"):
        snapshot = amrs[amrs["timestamp"] == timestamp]

        if snapshot.empty:
            continue

        eligible = prepare_eligible_amrs(snapshot, timestamp)
        if eligible.empty:
            continue

        window_end = timestamp + pd.Timedelta(minutes=WINDOW_MINUTES)
        window_requests = requests[
            (requests["request_timestamp"] > timestamp)
            & (requests["request_timestamp"] <= window_end)
        ].copy()

        # No-preposition baseline.
        assigned, response_distance = assign_requests_to_nearest_amrs(
            window_requests,
            eligible,
            station_lookup,
        )

        detail_rows.append(
            {
                "timestamp": timestamp,
                "strategy": "No pre-positioning",
                "requests_in_window": len(window_requests),
                "requests_assigned": assigned,
                "staging_distance_m": 0.0,
                "response_distance_m": response_distance,
            }
        )

        # Compare forecast-guided staging for each model.
        for model_name, prediction_column in MODEL_COLUMNS.items():
            if forecast_rows[prediction_column].isna().all():
                continue

            staged_positions, staging_distance = preposition_amrs(
                eligible,
                forecast_rows,
                prediction_column,
                station_lookup,
            )

            assigned, response_distance = assign_requests_to_nearest_amrs(
                window_requests,
                staged_positions,
                station_lookup,
            )

            detail_rows.append(
                {
                    "timestamp": timestamp,
                    "strategy": f"{model_name} pre-positioning",
                    "requests_in_window": len(window_requests),
                    "requests_assigned": assigned,
                    "staging_distance_m": staging_distance,
                    "response_distance_m": response_distance,
                }
            )

    details = pd.DataFrame(detail_rows)

    if details.empty:
        raise ValueError(
            "No evaluation windows were created. Check that forecast "
            "timestamps match the AMR snapshot timestamps."
        )

    details["total_distance_m"] = (
        details["staging_distance_m"] + details["response_distance_m"]
    )

    summary_rows = []

    for strategy, group in details.groupby("strategy"):
        total_requests = int(group["requests_in_window"].sum())
        total_assigned = int(group["requests_assigned"].sum())
        total_staging = group["staging_distance_m"].sum()
        total_response = group["response_distance_m"].sum()
        total_distance = group["total_distance_m"].sum()

        summary_rows.append(
            {
                "strategy": strategy,
                "windows_evaluated": group["timestamp"].nunique(),
                "requests_seen_in_windows": total_requests,
                "assignment_rate_pct": (
                    100 * total_assigned / total_requests
                    if total_requests
                    else 0.0
                ),
                "average_staging_distance_m_per_window": (
                    group["staging_distance_m"].mean()
                ),
                "response_distance_m_per_assigned_request": (
                    total_response / total_assigned
                    if total_assigned
                    else np.nan
                ),
                "total_distance_m_per_assigned_request": (
                    total_distance / total_assigned
                    if total_assigned
                    else np.nan
                ),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("strategy")

    OUTPUT_DIR.mkdir(exist_ok=True)
    detail_file = OUTPUT_DIR / "forecast_allocation_replay_by_window.csv"
    summary_file = OUTPUT_DIR / "forecast_allocation_replay_summary.csv"

    details.to_csv(detail_file, index=False)
    summary.to_csv(summary_file, index=False)

    print(f"Evaluation windows: {details['timestamp'].nunique()}")
    print(
        "Note: request windows overlap, so requests can appear in "
        "multiple evaluation windows."
    )
    print(
        "Distances are straight-line estimates; this does not model "
        "warehouse routes, travel time, task duration, or battery use."
    )
    print("\nAllocation replay summary:")
    print(summary.round(2).to_string(index=False))
    print(f"\nSaved detailed results to: {detail_file}")
    print(f"Saved summary to: {summary_file}")


if __name__ == "__main__":
    main()