from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_FILE = ROOT / "outputs" / "model_comparison_predictions.csv"

ACTUAL_COLUMN = "demand_next_15min"
PREDICTION_COLUMN = "xgboost_prediction"


def main():
    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Could not find:\n{PREDICTIONS_FILE}\n\n"
            "Run py src/train_xgboost.py first to create the test predictions."
        )

    data = pd.read_csv(PREDICTIONS_FILE)

    required_columns = [ACTUAL_COLUMN, PREDICTION_COLUMN]
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}\n"
            f"Columns in the file are: {data.columns.tolist()}"
        )

    actual_values = pd.to_numeric(data[ACTUAL_COLUMN], errors="coerce")
    predicted_values = pd.to_numeric(
        data[PREDICTION_COLUMN],
        errors="coerce",
    )

    valid_rows = actual_values.notna() & predicted_values.notna()
    actual = actual_values[valid_rows].to_numpy()
    predicted = predicted_values[valid_rows].to_numpy()

    if len(actual) == 0:
        raise ValueError("There are no valid actual and predicted values to compare.")

    errors = predicted - actual
    absolute_errors = np.abs(errors)

    mae = absolute_errors.mean()
    rmse = np.sqrt(np.mean(errors ** 2))
    average_bias = errors.mean()

    within_one_request_pct = (
        absolute_errors <= 1
    ).mean() * 100

    actual_total = np.abs(actual).sum()
    if actual_total == 0:
        wape = float("nan")
        wape_score = float("nan")
    else:
        wape = absolute_errors.sum() / actual_total * 100
        wape_score = 100 - wape

    print("XGBoost held-out test results")
    print(f"Prediction rows evaluated: {len(actual):,}")
    print(f"MAE: {mae:.3f} requests")
    print(f"RMSE: {rmse:.3f} requests")
    print(f"Average bias: {average_bias:.3f} requests")
    print(
        "Predictions within 1 request: "
        f"{within_one_request_pct:.1f}%"
    )
    print(f"WAPE: {wape:.1f}%")
    print(f"100% minus WAPE: {wape_score:.1f}%")
    print()
    print(f"Actual total across evaluated rows:    {actual.sum():.1f}")
    print(f"Predicted total across evaluated rows: {predicted.sum():.1f}")


if __name__ == "__main__":
    main()