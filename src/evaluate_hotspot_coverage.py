from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parent.parent
INPUT_FILE = ROOT / "outputs" / "stgt_predictions.csv"
OUTPUT_DIR = ROOT / "outputs"
TOP_K = 5

MODELS = {
    "STGT": "stgt_prediction",
    "XGBoost": "xgboost_prediction",
    "Baseline": "recent_demand_baseline",
}


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"File not found: {INPUT_FILE}")

    data = pd.read_csv(INPUT_FILE, parse_dates=["timestamp"])

    required_columns = [
        "timestamp",
        "station_id",
        "actual_demand",
        *MODELS.values(),
    ]
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise ValueError(f"Missing columns in the predictions file: {missing}")

    for column in ["actual_demand", *MODELS.values()]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["station_id"] = data["station_id"].astype(str)
    data = data.dropna(subset=required_columns).copy()

    # These final timestamps forecast beyond the 21:00 shift end.
    shift_end_padding = (
        (data["timestamp"].dt.hour == 20)
        & (data["timestamp"].dt.minute > 45)
    )
    excluded_rows = int(shift_end_padding.sum())
    data = data[~shift_end_padding].copy()

    details = []
    summary = []

    for model_name, prediction_column in MODELS.items():
        captured_total = 0.0
        actual_total = 0.0
        coverage_by_timestamp = []
        overlap_by_timestamp = []
        valid_timestamps = 0

        for timestamp, group in data.groupby("timestamp"):
            if len(group) < TOP_K:
                continue

            actual_demand_total = group["actual_demand"].sum()
            if actual_demand_total <= 0:
                continue

            predicted_top = group.nlargest(TOP_K, prediction_column)
            actual_top = group.nlargest(TOP_K, "actual_demand")

            captured = predicted_top["actual_demand"].sum()
            coverage = captured / actual_demand_total

            overlap_count = len(
                set(predicted_top["station_id"])
                & set(actual_top["station_id"])
            )

            captured_total += captured
            actual_total += actual_demand_total
            coverage_by_timestamp.append(coverage)
            overlap_by_timestamp.append(overlap_count / TOP_K)
            valid_timestamps += 1

            details.append(
                {
                    "timestamp": timestamp,
                    "model": model_name,
                    "actual_requests_total": actual_demand_total,
                    "actual_requests_at_predicted_top5": captured,
                    "percent_of_requests_at_predicted_top5": 100 * coverage,
                    "top5_overlap_with_actual_top5": overlap_count,
                }
            )

        if valid_timestamps:
            summary.append(
                {
                    "model": model_name,
                    "timestamps_evaluated": valid_timestamps,
                    "average_requests_at_predicted_top5": (
                        captured_total / valid_timestamps
                    ),
                    "overall_request_coverage_pct": (
                        100 * captured_total / actual_total
                    ),
                    "average_top5_overlap_pct": (
                        100 * sum(overlap_by_timestamp) / valid_timestamps
                    ),
                }
            )

    summary_df = pd.DataFrame(summary)
    details_df = pd.DataFrame(details)

    if summary_df.empty:
        raise ValueError("No timestamps with actual demand were available.")

    print(f"Timestamps evaluated: {summary_df['timestamps_evaluated'].min()}")
    print(f"Rows excluded at shift end: {excluded_rows}")
    print("\nHotspot coverage comparison:")
    print(summary_df.round(2).to_string(index=False))

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_file = OUTPUT_DIR / "hotspot_coverage_by_timestamp.csv"
    details_df.to_csv(output_file, index=False)
    print(f"\nSaved per-timestamp results to: {output_file}")


if __name__ == "__main__":
    main()