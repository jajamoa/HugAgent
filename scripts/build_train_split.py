#!/usr/bin/env python3
"""Build the train split from the synthetic track (Benchmark/synth_500.zip).

The synthetic participants are generated personas, not people, and are never
used for the paper's numbers. They are the only data meant for training.
The release ships no train split; this script builds one locally for anyone
who wants to train, in the Hub layout and in OdysSim's verl layout. Write it
outside hf/ so it is not uploaded. Test rows are never read here.

  python scripts/build_train_split.py --parquet-out /tmp/hugagent-train --odyssim-out /tmp/hugagent-train/odyssim
"""
import argparse, io, json, re, zipfile
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

TASKS = {"belief_attribution": "belief_state_inference", "belief_update": "belief_dynamics_update"}
SOURCES = {"belief_state_inference": "hugagent_bsi", "belief_dynamics_update": "hugagent_bdu"}
DOMAINS = ["healthcare", "surveillance", "zoning"]
HEX24 = re.compile(r"\b[0-9a-f]{24}\b")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")


def read_objects(text):
    """The synthetic files are pretty-printed JSON objects back to back."""
    dec, out, i = json.JSONDecoder(), [], 0
    while i < len(text):
        while i < len(text) and text[i] in " \r\n\t,":
            i += 1
        if i >= len(text):
            break
        obj, i = dec.raw_decode(text, i)
        out.append(obj)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, default=Path("Benchmark/synth_500.zip"))
    ap.add_argument("--parquet-out", type=Path, required=True)
    ap.add_argument("--odyssim-out", type=Path, required=True)
    args = ap.parse_args()

    zf = zipfile.ZipFile(args.zip)
    names = {n.split("/")[-1]: n for n in zf.namelist() if n.endswith(".jsonl") and "__MACOSX" not in n}
    pmap, public = {}, {}
    for task, pub_task in TASKS.items():
        rows = []
        for d in DOMAINS:
            raw = read_objects(zf.read(names[f"sample_{task}_{d}.jsonl"]).decode("utf-8"))
            for r in raw:
                pid = pmap.setdefault(r["prolific_id"], f"S{len(pmap) + 1:03d}")
                o = {"item_id": f"{pub_task}-{d}-train-{len(rows) + 1:05d}", "participant_id": pid, "split": "train"}
                for k, v in r.items():
                    if k not in ("id", "prolific_id"):
                        o[k] = v
                o["demographics"] = {k: v for k, v in o["demographics"].items() if k != "zipcode"}
                rows.append(o)
        blob = "\n".join(json.dumps(o, ensure_ascii=False) for o in rows)
        if HEX24.search(blob) or EMAIL.search(blob):
            raise SystemExit("privacy lint failed on the synthetic track")
        public[pub_task] = rows
        print(f"{pub_task}: {len(rows)} train rows, {len({o['participant_id'] for o in rows})} synthetic participants")

    for pub_task, rows in public.items():
        df = pd.DataFrame(rows)
        for col in ("demographics", "context_qas", "answer_options", "source_qa", "scale"):
            if col in df:
                df[col] = df[col].map(lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v)
        path = args.parquet_out / pub_task / "train-00000-of-00001.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        print(f"{path}: {len(df)} rows")
        table = pa.table({
            "prompt": pa.array([[{"content": "x", "role": "user"}]] * len(rows)),
            "data_source": pa.array([SOURCES[pub_task]] * len(rows)),
            "extra_info": pa.array(rows),
        })
        args.odyssim_out.mkdir(parents=True, exist_ok=True)
        opath = args.odyssim_out / f"{SOURCES[pub_task]}_train.parquet"
        pq.write_table(table, opath)
        print(f"{opath}: {len(rows)} rows")


if __name__ == "__main__":
    main()
