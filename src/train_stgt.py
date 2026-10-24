from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "AMR_FINAL_30DAY_RESEARCH_READY"
OUTPUT_DIR = ROOT / "outputs"

STATE_FILE = DATA_DIR / "station_state_5min.csv"
STATION_FILE = DATA_DIR / "station_master.csv"
GRAPH_FILE = OUTPUT_DIR / "station_graph_edges.csv"
XGB_FILE = OUTPUT_DIR / "model_comparison_predictions.csv"

TARGET = "demand_next_15min"
LOOKBACK = 12
BATCH_SIZE = 32
EPOCHS = 25
DEVICE = torch.device("cpu")

torch.manual_seed(42)
np.random.seed(42)

# Load stations and keep a consistent node order.
stations = pd.read_csv(STATION_FILE)
stations["station_id"] = stations["station_id"].astype(str)
stations = stations.sort_values("station_id").reset_index(drop=True)

station_ids = stations["station_id"].tolist()
n_stations = len(station_ids)

# Load station-state data.
df = pd.read_csv(STATE_FILE, parse_dates=["timestamp"])
df["station_id"] = df["station_id"].astype(str)
df = df.sort_values(["timestamp", "station_id"]).reset_index(drop=True)

if not GRAPH_FILE.exists():
    raise FileNotFoundError(
        f"Graph file not found: {GRAPH_FILE}\n"
        "Run this first: py src/build_station_graph.py"
    )

# Build a weighted adjacency matrix from station_graph_edges.csv.
edge_df = pd.read_csv(GRAPH_FILE)
station_index = {station_id: i for i, station_id in enumerate(station_ids)}

adjacency = np.zeros((n_stations, n_stations), dtype=np.float32)
np.fill_diagonal(adjacency, 1.0)

for row in edge_df.itertuples(index=False):
    from_id = str(row.from_station)
    to_id = str(row.to_station)

    if from_id not in station_index or to_id not in station_index:
        raise ValueError(f"Graph contains unknown station: {from_id} or {to_id}")

    i = station_index[from_id]
    j = station_index[to_id]
    weight = float(row.weight)

    adjacency[i, j] = weight
    adjacency[j, i] = weight

# Connected stations can attend to each other. Every station can also
# attend to itself.
adjacency_tensor = torch.tensor(adjacency, dtype=torch.float32)
graph_bias = torch.full_like(adjacency_tensor, float("-inf"))
connected = adjacency_tensor > 0
graph_bias[connected] = torch.log(adjacency_tensor[connected])

# Station-state features available at prediction time.
numeric_features = [
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

required_columns = set(numeric_features + [TARGET, "zone"])
missing_columns = required_columns - set(df.columns)
if missing_columns:
    raise ValueError(
        f"station_state_5min.csv is missing: {sorted(missing_columns)}"
    )

for column in numeric_features + [TARGET]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# One timestamp per model step, in chronological order.
times = pd.DatetimeIndex(sorted(df["timestamp"].unique()))
if len(times) < LOOKBACK:
    raise ValueError(f"Need at least {LOOKBACK} timestamps; found {len(times)}.")

def make_station_matrix(column):
    """Make a [time, station] array in station_master order."""
    wide = df.pivot(
        index="timestamp",
        columns="station_id",
        values=column,
    ).reindex(index=times, columns=station_ids)

    return wide.to_numpy(dtype=np.float32)

# Numeric inputs: [time, station, feature].
feature_matrices = []
for column in numeric_features:
    feature_matrices.append(make_station_matrix(column)[:, :, None])

X_numeric = np.concatenate(feature_matrices, axis=2)

# Static station-zone one-hot features, repeated over time.
zone_by_station = stations.set_index("station_id").loc[station_ids, "zone"]
zone_one_hot = pd.get_dummies(zone_by_station).to_numpy(dtype=np.float32)
X_zone = np.broadcast_to(
    zone_one_hot[None, :, :],
    (len(times), n_stations, zone_one_hot.shape[1]),
)

# Time-of-day and day-of-week features.
hours = np.array(
    [timestamp.hour + timestamp.minute / 60 for timestamp in times],
    dtype=np.float32,
)
weekdays = np.array([timestamp.dayofweek for timestamp in times], dtype=np.float32)

time_features = np.column_stack(
    [
        np.sin(2 * np.pi * hours / 24),
        np.cos(2 * np.pi * hours / 24),
        np.sin(2 * np.pi * weekdays / 7),
        np.cos(2 * np.pi * weekdays / 7),
    ]
).astype(np.float32)

X_time = np.broadcast_to(
    time_features[:, None, :],
    (len(times), n_stations, time_features.shape[1]),
)

X = np.concatenate([X_numeric, X_zone, X_time], axis=2).astype(np.float32)
Y = make_station_matrix(TARGET)
baseline = make_station_matrix("recent_demand_15min")

# Chronological splits.
train_boundary = pd.Timestamp(times[int(len(times) * 0.70)])
test_start = pd.Timestamp(times[int(len(times) * 0.85)])

# A target for time t is not known until the following 15 minutes have passed.
label_cutoff = train_boundary - pd.Timedelta(minutes=15)

# Normalize timestamps to nanoseconds before checking 5-minute gaps.
time_ns = times.as_unit("ns").asi8
five_minutes_ns = 5 * 60 * 1_000_000_000
label_cutoff_ns = label_cutoff.as_unit("ns").value

# Fit feature imputation and scaling on training-period rows only.
cut_index = int(np.searchsorted(time_ns, label_cutoff_ns, side="left"))
if cut_index == 0:
    raise ValueError("There are no timestamps before the training cutoff.")

medians = np.nanmedian(X[:cut_index], axis=(0, 1))
medians = np.where(np.isfinite(medians), medians, 0.0)

for feature_index in range(X.shape[2]):
    values = X[:, :, feature_index]
    X[:, :, feature_index] = np.where(
        np.isfinite(values),
        values,
        medians[feature_index],
    )

means = X[:cut_index].mean(axis=(0, 1))
scales = X[:cut_index].std(axis=(0, 1))
scales = np.where(scales > 1e-6, scales, 1.0)
X = ((X - means) / scales).astype(np.float32)

# Make one-hour windows only when all 12 timestamps are exactly 5 minutes apart.
train_ends = []
validation_ends = []
test_ends = []

for end in range(LOOKBACK - 1, len(times)):
    start = end - LOOKBACK + 1
    window_differences = np.diff(time_ns[start : end + 1])

    if not np.all(window_differences == five_minutes_ns):
        continue
    if not np.isfinite(Y[end]).all():
        continue

    timestamp = pd.Timestamp(times[end])

    if timestamp < label_cutoff:
        train_ends.append(end)
    elif train_boundary <= timestamp < test_start:
        validation_ends.append(end)
    elif timestamp >= test_start:
        test_ends.append(end)

print(f"Unique timestamps: {len(times)}")
print(f"5-minute intervals: {int(np.sum(np.diff(time_ns) == five_minutes_ns))}")
print(f"Training windows: {len(train_ends)}")
print(f"Validation windows: {len(validation_ends)}")
print(f"Test windows: {len(test_ends)}")
print(f"Training cutoff: {label_cutoff}")
print(f"Test starts: {test_start}")

if not train_ends or not validation_ends or not test_ends:
    raise ValueError(
        "At least one split has no usable sequences. "
        "The counts printed above show which split is empty."
    )

class WindowDataset(Dataset):
    def __init__(self, features, targets, end_indices):
        self.features = features
        self.targets = targets
        self.end_indices = end_indices

    def __len__(self):
        return len(self.end_indices)

    def __getitem__(self, item):
        end = self.end_indices[item]
        start = end - LOOKBACK + 1

        # copy() makes these arrays writable before PyTorch converts them.
        x_window = self.features[start : end + 1].copy()
        y_at_end = self.targets[end].copy()

        return torch.from_numpy(x_window), torch.from_numpy(y_at_end)

train_loader = DataLoader(
    WindowDataset(X, Y, train_ends),
    batch_size=BATCH_SIZE,
    shuffle=True,
)
validation_loader = DataLoader(
    WindowDataset(X, Y, validation_ends),
    batch_size=BATCH_SIZE,
    shuffle=False,
)
test_loader = DataLoader(
    WindowDataset(X, Y, test_ends),
    batch_size=BATCH_SIZE,
    shuffle=False,
)

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

        # Attend over the past time steps for each station.
        h = h.reshape(batch, steps, station_count, -1)
        h = h.permute(0, 2, 1, 3).reshape(batch * station_count, steps, -1)
        h = h + self.position[:, :steps, :]
        h = self.temporal_transformer(h)

        # Predict demand_next_15min for every station.
        latest = h[:, -1, :]
        prediction = self.output_layer(latest)
        return prediction.reshape(batch, station_count)

model = SpatioTemporalGraphTransformer(
    feature_count=X.shape[2],
    lookback=LOOKBACK,
    graph_attention_bias=graph_bias,
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.MSELoss()

def predict(loader):
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(DEVICE)
            predictions.append(model(x_batch).cpu().numpy())
            actuals.append(y_batch.numpy())

    return np.concatenate(predictions), np.concatenate(actuals)

best_validation_mae = float("inf")
best_state = None
patience = 6
epochs_without_improvement = 0

print(f"\nStations: {n_stations}")
print(f"Graph connections: {len(edge_df)}")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()
        predictions = model(x_batch)
        loss = loss_function(predictions, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    validation_predictions, validation_actuals = predict(validation_loader)
    validation_mae = mean_absolute_error(
        validation_actuals.reshape(-1),
        validation_predictions.reshape(-1),
    )

    print(
        f"Epoch {epoch:02d}/{EPOCHS} "
        f"train_loss={total_loss / len(train_loader):.4f} "
        f"validation_MAE={validation_mae:.3f}"
    )

    if validation_mae < best_validation_mae:
        best_validation_mae = validation_mae
        best_state = deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print("Stopping because validation MAE stopped improving.")
            break

if best_state is None:
    raise RuntimeError("Training did not produce a model checkpoint.")

model.load_state_dict(best_state)
test_predictions, test_actuals = predict(test_loader)
test_predictions = np.maximum(test_predictions, 0)

# Save one row per test timestamp and workstation.
prediction_rows = []

for row_index, end in enumerate(test_ends):
    for station_index, station_id in enumerate(station_ids):
        prediction_rows.append(
            {
                "timestamp": times[end],
                "station_id": station_id,
                "actual_demand": float(test_actuals[row_index, station_index]),
                "stgt_prediction": float(
                    test_predictions[row_index, station_index]
                ),
                "recent_demand_baseline": float(
                    max(0.0, baseline[end, station_index])
                ),
            }
        )

results = pd.DataFrame(prediction_rows)

def print_scores(label, actual, prediction):
    mae = mean_absolute_error(actual, prediction)
    rmse = root_mean_squared_error(actual, prediction)
    print(f"  {label:8s} MAE={mae:.3f}, RMSE={rmse:.3f}")

print("\nTest results on all usable STGT timestamps:")
print_scores(
    "STGT",
    results["actual_demand"],
    results["stgt_prediction"],
)
print_scores(
    "Baseline",
    results["actual_demand"],
    results["recent_demand_baseline"],
)

# Compare XGBoost on the exact same timestamps.
if XGB_FILE.exists():
    xgb_results = pd.read_csv(XGB_FILE, parse_dates=["timestamp"])
    xgb_results["station_id"] = xgb_results["station_id"].astype(str)

    results = results.merge(
        xgb_results[["timestamp", "station_id", "xgboost_prediction"]],
        on=["timestamp", "station_id"],
        how="left",
    )

    common_xgb = results.dropna(subset=["xgboost_prediction"])
    if not common_xgb.empty:
        print_scores(
            "XGBoost",
            common_xgb["actual_demand"],
            common_xgb["xgboost_prediction"],
        )

OUTPUT_DIR.mkdir(exist_ok=True)
results.to_csv(OUTPUT_DIR / "stgt_predictions.csv", index=False)

torch.save(
    {
        "state_dict": model.state_dict(),
        "station_ids": station_ids,
        "lookback": LOOKBACK,
        "feature_count": X.shape[2],
        "feature_medians": medians.tolist(),
        "feature_means": means.tolist(),
        "feature_scales": scales.tolist(),
    },
    OUTPUT_DIR / "stgt_model.pth",
)

print(f"\nSaved STGT predictions and model in: {OUTPUT_DIR}")
