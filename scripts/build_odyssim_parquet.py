#!/usr/bin/env python3
"""Write the HugAgent items in the verl row format that OdysSim's eval.sh reads.

Each row: prompt=[{"role": "user", "content": "x"}] (placeholder, the agent builds
the real prompt), data_source (hugagent_bsi or hugagent_bdu), extra_info (the item).

  python scripts/build_odyssim_parquet.py --data-dir Benchmark/data --out hf/odyssim
"""
import argparse, json
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

DOMAINS = ["healthcare", "surveillance", "zoning"]
TASKS = {"belief_state_inference": "hugagent_bsi", "belief_dynamics_update": "hugagent_bdu"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("Benchmark/data"))
    ap.add_argument("--out", type=Path, default=Path("hf/odyssim"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for task, source in TASKS.items():
        rows = []
        for d in DOMAINS:
            for line in (args.data_dir / f"{task}_{d}.jsonl").read_text().splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        table = pa.table({
            "prompt": pa.array([[{"content": "x", "role": "user"}]] * len(rows)),
            "data_source": pa.array([source] * len(rows)),
            "extra_info": pa.array(rows),
        })
        path = args.out / f"{source}_val.parquet"
        pq.write_table(table, path)
        print(f"{path}: {len(rows)} rows")


if __name__ == "__main__":
    main()
