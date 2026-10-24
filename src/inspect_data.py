import csv
from pathlib import Path

file_path = (
    Path(__file__).parent.parent
    / "data"
    / "AMR_FINAL_30DAY_RESEARCH_READY"
    / "station_state_5min.csv"
)

with file_path.open("r", newline="", encoding="utf-8-sig") as file:
    reader = csv.reader(file)
    headers = next(reader)

    print("Columns:")
    for column in headers:
        print("-", column)

    print("\nFirst 3 rows:")
    for row in list(reader)[:3]:
        print(row)