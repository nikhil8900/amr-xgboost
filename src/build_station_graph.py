from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
STATION_FILE = (
    ROOT / "data" / "AMR_FINAL_30DAY_RESEARCH_READY" / "station_master.csv"
)
OUTPUT_FILE = ROOT / "outputs" / "station_graph_edges.csv"

stations = pd.read_csv(STATION_FILE).sort_values("station_id").reset_index(drop=True)

required = {"station_id", "x_m", "y_m"}
missing = required - set(stations.columns)
if missing:
    raise ValueError(f"station_master.csv is missing columns: {sorted(missing)}")

coordinates = stations[["x_m", "y_m"]].to_numpy(dtype=float)
station_ids = stations["station_id"].tolist()

# Euclidean distances between every pair of workstations.
difference = coordinates[:, None, :] - coordinates[None, :, :]
distances = np.sqrt((difference ** 2).sum(axis=2))

# Connect each workstation to its four nearest neighbors.
# Store each connection once; these edges are treated as undirected.
edges = set()
neighbors_per_station = min(4, len(station_ids) - 1)

for i in range(len(station_ids)):
    nearest = np.argsort(distances[i])[1 : neighbors_per_station + 1]
    for j in nearest:
        edges.add(tuple(sorted((i, int(j)))))

edge_rows = []
edge_distances = []

for i, j in sorted(edges):
    distance = float(distances[i, j])
    edge_distances.append(distance)
    edge_rows.append(
        {
            "from_station": station_ids[i],
            "to_station": station_ids[j],
            "distance_m": distance,
        }
    )

# Convert distance to a weight between 0 and 1.
scale = float(np.median(edge_distances)) if edge_distances else 1.0
for row in edge_rows:
    row["weight"] = float(np.exp(-((row["distance_m"] / scale) ** 2)))

OUTPUT_FILE.parent.mkdir(exist_ok=True)
pd.DataFrame(edge_rows).to_csv(OUTPUT_FILE, index=False)

print(f"Stations: {len(station_ids)}")
print(f"Undirected connections: {len(edge_rows)}")
print(f"Saved graph edges to: {OUTPUT_FILE}")
print("\nFirst connections:")
print(pd.DataFrame(edge_rows).head(10).to_string(index=False))