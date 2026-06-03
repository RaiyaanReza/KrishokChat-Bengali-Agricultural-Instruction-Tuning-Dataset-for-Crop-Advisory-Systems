import json
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "dataset_hf"

JSONL_FILES = [
    "train.jsonl",
    "val.jsonl",
    "test.jsonl",
    "full_dataset.jsonl",
    "instructions_unified.jsonl",
    "safety_qas.jsonl",
]

JSON_FILES = [
    "farmer_benchmark.json",
    "knowledge_nodes.json",
]


def jsonl_to_parquet(jsonl_path: Path):
    parquet_path = jsonl_path.with_suffix(".parquet")
    if parquet_path.exists():
        print(f"  Skipping {parquet_path.name} (already exists)")
        return
    print(f"  Converting {jsonl_path.name} -> {parquet_path.name} ...")
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, parquet_path, compression="snappy")
    print(f"    Done: {len(rows)} rows, {parquet_path.stat().st_size / 1024**2:.1f} MB")


def json_to_parquet(json_path: Path):
    parquet_path = json_path.with_suffix(".parquet")
    if parquet_path.exists():
        print(f"  Skipping {parquet_path.name} (already exists)")
        return
    print(f"  Converting {json_path.name} -> {parquet_path.name} ...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = [data]
    else:
        raise ValueError(f"Unexpected JSON structure in {json_path.name}")
    df = pd.DataFrame(rows)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, parquet_path, compression="snappy")
    print(f"    Done: {len(rows)} rows, {parquet_path.stat().st_size / 1024**2:.1f} MB")


def main():
    print("Converting JSONL files to Parquet...")
    for fname in JSONL_FILES:
        path = BASE / fname
        if path.exists():
            jsonl_to_parquet(path)
        else:
            print(f"  Skipping {fname} (not found)")

    print("Converting JSON files to Parquet...")
    for fname in JSON_FILES:
        path = BASE / fname
        if path.exists():
            json_to_parquet(path)
        else:
            print(f"  Skipping {fname} (not found)")

    print("All conversions complete.")


if __name__ == "__main__":
    main()
