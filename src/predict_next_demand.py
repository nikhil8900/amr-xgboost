from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "AMR_FINAL_30DAY_RESEARCH_READY"
OUTPUT_DIR = ROOT / "outputs"

STATE_FILE = DATA_DIR / "station_state_5min.csv"
STATION_FILE = DATA_DIR / "station_master.csv"
GRAPH_FILE = OUTPUT_DIR / "station_graph_edges.csv"
MODEL_FILE = OUTPUT_DIR / "stgt_model.pth"

TARGET = "demand_next_15min"
LOOKBACK = 12
DEVICE = torch.device("cpu")


# These must stay in the same order as in train_stgt.py.
NUMERIC_FEATURES = [
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
]


class SpatioTemporalGraphTransformer(nn.Module):
    def __init__(self, feature_count, lookback, graph_attention_bias):
        super().__init__()
        model_width = 64

        self.input_projection = nn.Linear(feature_count, model_width)
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=model_width,
            num_heads=4,
            batch_first=True,
        )
        self.spatial_norm = nn.LayerNorm(model_width)
        self.position = nn.Parameter(torch.zeros(1, lookback, model_width))

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=model_width,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            temporal_layer,
            num_layers=2,
        )
        self.output_layer = nn.Linear(model_width, 1)

        self.register_buffer("graph_bias", graph_attention_bias)

    def forward(self, x):
        batch, steps, station_count, _ = x.shape

        # Attend across connected stations at each time step.
        h = self.input_projection(x)
        h = h.reshape(batch * steps, station_count, -1)

        spatial, _ = self.spatial_attention(
            h,
            h,
            h,
            attn_mask=self.graph_bias,
            need_weights=False,
        )
        h = self.spatial_norm(h + spatial)

        # Attend over time separately for each station.
        h = h.reshape(batch, steps, station_count, -1)
        h = h.permute(0, 2, 1, 3).reshape(
            batch * station_count,
            steps,
            -1,
        )
        h = h + self.position[:, :steps, :]
        h = self.temporal_transformer(h)

        latest = h[:, -1, :]
        prediction = self.output_layer(latest)
        return prediction.reshape(batch, station_count)


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")


def main():
    require_file(STATE_FILE)
    require_file(STATION_FILE)
    require_file(GRAPH_FILE)
    require_file(MODEL_FILE)

    # Load the saved training information.
    checkpoint = torch.load(
        MODEL_FILE,
        map_location=DEVICE,
        weights_only=False,
    )

    station_ids = [str(station_id) for station_id in checkpoint["station_ids"]]
    lookback = int(checkpoint["lookback"])
    feature_count = int(checkpoint["feature_count"])

    if lookback != LOOKBACK:
        raise ValueError(
            f"The model expects {lookback} time steps, "
            f"but this script is configured for {LOOKBACK}."
        )

    # Load station information and keep the training station order.
    stations = pd.read_csv(STATION_FILE)
    stations["station_id"] = stations["station_id"].astype(str)
    stations = stations.set_index("station_id").reindex(station_ids)

    if stations["zone"].isna().any():
        raise ValueError(
            "Some model stations are missing from station_master.csv."
        )

    # Load recent station data.
    df = pd.read_csv(STATE_FILE, parse_dates=["timestamp"])
    df["station_id"] = df["station_id"].astype(str)
    df = df.sort_values(["timestamp", "station_id"]).reset_index(drop=True)

    missing = set(NUMERIC_FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Data is missing required columns: {sorted(missing)}")

    for column in NUMERIC_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    times = pd.DatetimeIndex(sorted(df["timestamp"].dropna().unique()))

    if len(times) < LOOKBACK:
        raise ValueError(
            f"Need at least {LOOKBACK} timestamps; found {len(times)}."
        )

    latest_times = times[-LOOKBACK:]
    differences = np.diff(latest_times.as_unit("ns").asi8)
    five_minutes_ns = 5 * 60 * 1_000_000_000

    if not np.all(differences == five_minutes_ns):
        raise ValueError(
            "The latest 12 timestamps are not continuous five-minute "
            "intervals. Check the most recent rows in station_state_5min.csv."
        )

    # Create [time, station] matrices in the saved model's station order.
    def make_station_matrix(column):
        wide = df.pivot(
            index="timestamp",
            columns="station_id",
            values=column,
        ).reindex(index=latest_times, columns=station_ids)

        return wide.to_numpy(dtype=np.float32)

    # Assemble numeric input features.
    numeric_matrices = [
        make_station_matrix(column)[:, :, None]
        for column in NUMERIC_FEATURES
    ]
    x_numeric = np.concatenate(numeric_matrices, axis=2)

    # Add station-zone one-hot features, using the same approach as training.
    zone_by_station = stations["zone"]
    zone_one_hot = pd.get_dummies(zone_by_station).to_numpy(dtype=np.float32)

    x_zone = np.broadcast_to(
        zone_one_hot[None, :, :],
        (LOOKBACK, len(station_ids), zone_one_hot.shape[1]),
    )

    # Add time of day and day of week features.
    hours = np.array(
        [timestamp.hour + timestamp.minute / 60 for timestamp in latest_times],
        dtype=np.float32,
    )
    weekdays = np.array(
        [timestamp.dayofweek for timestamp in latest_times],
        dtype=np.float32,
    )

    time_features = np.column_stack(
        [
            np.sin(2 * np.pi * hours / 24),
            np.cos(2 * np.pi * hours / 24),
            np.sin(2 * np.pi * weekdays / 7),
            np.cos(2 * np.pi * weekdays / 7),
        ]
    ).astype(np.float32)

    x_time = np.broadcast_to(
        time_features[:, None, :],
        (LOOKBACK, len(station_ids), time_features.shape[1]),
    )

    x = np.concatenate([x_numeric, x_zone, x_time], axis=2).astype(np.float32)

    if x.shape[2] != feature_count:
        raise ValueError(
            f"Feature count mismatch: model expects {feature_count}, "
            f"but this script created {x.shape[2]}."
        )

    # Apply the same imputation and scaling values used during training.
    medians = np.asarray(checkpoint["feature_medians"], dtype=np.float32)
    means = np.asarray(checkpoint["feature_means"], dtype=np.float32)
    scales = np.asarray(checkpoint["feature_scales"], dtype=np.float32)

    for feature_index in range(x.shape[2]):
        values = x[:, :, feature_index]
        x[:, :, feature_index] = np.where(
            np.isfinite(values),
            values,
            medians[feature_index],
        )

    x = ((x - means) / scales).astype(np.float32)

    # Rebuild the station graph attention mask.
    station_index = {
        station_id: index for index, station_id in enumerate(station_ids)
    }
    adjacency = np.zeros((len(station_ids), len(station_ids)), dtype=np.float32)
    np.fill_diagonal(adjacency, 1.0)

    edge_df = pd.read_csv(GRAPH_FILE)
    for row in edge_df.itertuples(index=False):
        from_station = str(row.from_station)
        to_station = str(row.to_station)

        if from_station not in station_index or to_station not in station_index:
            raise ValueError(
                f"Graph contains an unknown station: "
                f"{from_station} or {to_station}"
            )

        i = station_index[from_station]
        j = station_index[to_station]
        weight = float(row.weight)
        adjacency[i, j] = weight
        adjacency[j, i] = weight

    adjacency_tensor = torch.tensor(adjacency, dtype=torch.float32)
    graph_bias = torch.full_like(adjacency_tensor, float("-inf"))
    connected = adjacency_tensor > 0
    graph_bias[connected] = torch.log(adjacency_tensor[connected])

    # Create the model and load its trained weights.
    model = SpatioTemporalGraphTransformer(
        feature_count=feature_count,
        lookback=lookback,
        graph_attention_bias=graph_bias,
    ).to(DEVICE)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Predict demand for all stations from the latest 12 records.
    x_tensor = torch.from_numpy(x[None, ...].copy()).to(DEVICE)

    with torch.no_grad():
        predictions = model(x_tensor).cpu().numpy().reshape(-1)

    predictions = np.maximum(predictions, 0)

    # Save one forecast row per workstation.
    results = pd.DataFrame(
        {
            "input_timestamp": [latest_times[-1]] * len(station_ids),
            "station_id": station_ids,
            "zone": stations["zone"].to_numpy(),
            "predicted_demand_next_15min": predictions,
        }
    )

    results = results.sort_values(
        "predicted_demand_next_15min",
        ascending=False,
    ).reset_index(drop=True)

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_file = OUTPUT_DIR / "next_demand_forecast.csv"
    results.to_csv(output_file, index=False)

    print(f"Used data through: {latest_times[-1]}")
    print("Forecast target: demand_next_15min")
    print("\nHighest predicted-demand stations:")
    print(results.head(10).to_string(index=False))
    print(f"\nSaved all station forecasts to: {output_file}")


if __name__ == "__main__":
    main()