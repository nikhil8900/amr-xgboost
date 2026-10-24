from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"

XGBOOST_FILE = OUTPUT_DIR / "model_comparison_predictions.csv"
STGT_FILE = OUTPUT_DIR / "stgt_predictions.csv"

TIMESTAMP = "timestamp"
STATION = "station_id"
TARGET = "demand_next_15min"
XGBOOST_PREDICTION = "xgboost_prediction"
STGT_PREDICTION = "stgt_prediction"


def main():
    for path in (XGBOOST_FILE, STGT_FILE):
        if not path.exists():
            raise FileNotFoundError(
                f"Could not find {path}\n"
                "Run the XGBoost and STGT training scripts first."
            )

    xgb = pd.read_csv(XGBOOST_FILE, parse_dates=[TIMESTAMP])
    stgt = pd.read_csv(STGT_FILE, parse_dates=[TIMESTAMP])

    required_xgb = {TIMESTAMP, STATION, TARGET, XGBOOST_PREDICTION}
    required_stgt = {TIMESTAMP, STATION, STGT_PREDICTION}

    if not required_xgb.issubset(xgb.columns):
        raise ValueError(
            f"{XGBOOST_FILE.name} needs these columns: "
            f"{sorted(required_xgb)}"
        )
    if not required_stgt.issubset(stgt.columns):
        raise ValueError(
            f"{STGT_FILE.name} needs these columns: "
            f"{sorted(required_stgt)}"
        )

    xgb[STATION] = xgb[STATION].astype(str)
    stgt[STATION] = stgt[STATION].astype(str)

    xgb = xgb[[TIMESTAMP, STATION, TARGET, XGBOOST_PREDICTION]].copy()
    stgt = stgt[[TIMESTAMP, STATION, STGT_PREDICTION]].copy()

    if xgb.duplicated([TIMESTAMP, STATION]).any():
        raise ValueError("XGBoost file has duplicate station/timestamp rows.")
    if stgt.duplicated([TIMESTAMP, STATION]).any():
        raise ValueError("STGT file has duplicate station/timestamp rows.")

    # Keep only station/timestamp pairs predicted by both models.
    data = xgb.merge(
        stgt,
        on=[TIMESTAMP, STATION],
        how="inner",
        validate="one_to_one",
    )

    for column in [TARGET, XGBOOST_PREDICTION, STGT_PREDICTION]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(
        subset=[TARGET, XGBOOST_PREDICTION, STGT_PREDICTION]
    ).copy()

    timestamps = sorted(data[TIMESTAMP].unique())
    if len(timestamps) < 10:
        raise ValueError(
            f"Only {len(timestamps)} shared timestamps were found. "
            "There are not enough for a useful blend evaluation."
        )

    # Tune the blend on earlier shared timestamps and reserve later ones
    # for evaluating the chosen blend.
    split_index = int(len(timestamps) * 0.70)
    split_index = min(max(split_index, 1), len(timestamps) - 1)

    validation_timestamps = set(timestamps[:split_index])
    test_timestamps = set(timestamps[split_index:])

    validation = data[data[TIMESTAMP].isin(validation_timestamps)].copy()
    test = data[data[TIMESTAMP].isin(test_timestamps)].copy()

    # weight = 1.0 means all XGBoost; 0.0 means all STGT.
    weight_results = []
    best_weight = None
    best_score = (float("inf"), float("inf"))

    for xgb_weight in np.round(np.arange(0.0, 1.01, 0.1), 1):
        blend = (
            xgb_weight * validation[XGBOOST_PREDICTION]
            + (1.0 - xgb_weight) * validation[STGT_PREDICTION]
        ).clip(lower=0)

        mae = mean_absolute_error(validation[TARGET], blend)
        rmse = root_mean_squared_error(validation[TARGET], blend)

        weight_results.append(
            {
                "xgboost_weight": xgb_weight,
                "stgt_weight": round(1.0 - xgb_weight, 1),
                "validation_mae": mae,
                "validation_rmse": rmse,
            }
        )

        # If scores tie, prefer the larger XGBoost weight.
        score = (mae, -xgb_weight)
        if score < best_score:
            best_score = score
            best_weight = xgb_weight

    weight_results = pd.DataFrame(weight_results)
    weight_results.to_csv(
        OUTPUT_DIR / "ensemble_weight_validation.csv",
        index=False,
    )

    test = test.copy()
    test["ensemble_prediction"] = (
        best_weight * test[XGBOOST_PREDICTION]
        + (1.0 - best_weight) * test[STGT_PREDICTION]
    ).clip(lower=0)

    def top_five_metrics(frame, prediction_column):
        overlaps = []
        coverage_values = []

        for _, group in frame.groupby(TIMESTAMP):
            k = min(5, len(group))
            predicted_top = set(
                group.nlargest(k, prediction_column)[STATION]
            )
            actual_top = set(group.nlargest(k, TARGET)[STATION])

            overlaps.append(len(predicted_top & actual_top) / k)

            total_actual = group[TARGET].sum()
            if total_actual > 0:
                covered_actual = group.loc[
                    group[STATION].isin(predicted_top), TARGET
                ].sum()
                coverage_values.append(covered_actual / total_actual)

        overlap_pct = 100 * float(np.mean(overlaps)) if overlaps else float("nan")
        coverage_pct = (
            100 * float(np.mean(coverage_values))
            if coverage_values
            else float("nan")
        )
        return overlap_pct, coverage_pct

    rows = []
    for label, prediction_column in [
        ("XGBoost", XGBOOST_PREDICTION),
        ("STGT", STGT_PREDICTION),
        ("Ensemble", "ensemble_prediction"),
    ]:
        mae = mean_absolute_error(test[TARGET], test[prediction_column])
        rmse = root_mean_squared_error(test[TARGET], test[prediction_column])
        overlap, coverage = top_five_metrics(test, prediction_column)

        rows.append(
            {
                "model": label,
                "test_timestamps": test[TIMESTAMP].nunique(),
                "test_rows": len(test),
                "MAE": mae,
                "RMSE": rmse,
                "average_top5_overlap_pct": overlap,
                "average_top5_demand_coverage_pct": coverage,
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(
        OUTPUT_DIR / "ensemble_test_summary.csv",
        index=False,
    )
    test.to_csv(
        OUTPUT_DIR / "ensemble_test_predictions.csv",
        index=False,
    )

    print(f"Shared timestamps: {len(timestamps)}")
    print(f"Blend-selection timestamps: {len(validation_timestamps)}")
    print(f"Final evaluation timestamps: {len(test_timestamps)}")
    print(
        f"\nSelected blend: {best_weight:.0%} XGBoost + "
        f"{1.0 - best_weight:.0%} STGT"
    )
    print("\nResults on later held-out timestamps:")
    print(summary.round(3).to_string(index=False))
    print("\nSaved:")
    print(f"  {OUTPUT_DIR / 'ensemble_weight_validation.csv'}")
    print(f"  {OUTPUT_DIR / 'ensemble_test_summary.csv'}")
    print(f"  {OUTPUT_DIR / 'ensemble_test_predictions.csv'}")


if __name__ == "__main__":
    main()