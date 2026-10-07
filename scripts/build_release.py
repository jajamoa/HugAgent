#!/usr/bin/env python3
"""Build the public HugAgent release from the internal question set.

Input: the six internal jsonl files (one per task x domain).
Output: six renamed jsonl files for GitHub, two parquet configs for the
Hugging Face Hub, and a privacy lint report. The participant mapping is
written outside the repository and must stay private.

Usage:
  python scripts/build_release.py --src <dir with internal jsonl> \
      --jsonl-out Benchmark/data --parquet-out hf/data \
      --map-out /private/path/participant_map.json
"""
import argparse, json, re, sys
from pathlib import Path

TASKS = {
    "belief_attribution": "belief_state_inference",
    "belief_update": "belief_dynamics_update",
}
DOMAINS = ["healthcare", "surveillance", "zoning"]

HEX24 = re.compile(r"\b[0-9a-f]{24}\b")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE = re.compile(r"\b\d{3}[-. ]\d{3}[-. ]\d{4}\b")


def find_source(src: Path, task: str, domain: str) -> Path:
    hits = sorted(src.glob(f"*{task}_{domain}*.jsonl"))
    if len(hits) != 1:
        sys.exit(f"expected one source file for {task}/{domain}, got {hits}")
    return hits[0]


def load(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--jsonl-out", required=True, type=Path)
    ap.add_argument("--parquet-out", type=Path)
    ap.add_argument("--map-out", required=True, type=Path,
                    help="private file: internal id prefix -> public participant_id")
    ap.add_argument("--keep-zip", action="store_true",
                    help="keep the zipcode field (default: drop it)")
    args = ap.parse_args()

    sources = {(t, d): find_source(args.src, t, d) for t in TASKS for d in DOMAINS}
    rows = {k: load(p) for k, p in sources.items()}

    # Stable participant ids: sort the internal prefixes once, number them.
    prefixes = sorted({r["prolific_id"] for rs in rows.values() for r in rs})
    pmap = {p: f"P{i + 1:02d}" for i, p in enumerate(prefixes)}
    args.map_out.parent.mkdir(parents=True, exist_ok=True)
    args.map_out.write_text(json.dumps(pmap, indent=2))

    args.jsonl_out.mkdir(parents=True, exist_ok=True)
    public = {}
    for (task, domain), rs in rows.items():
        pub_task = TASKS[task]
        out = []
        for i, r in enumerate(rs, start=1):
            o = {"item_id": f"{pub_task}-{domain}-{i:04d}",
                 "participant_id": pmap[r["prolific_id"]]}
            for k, v in r.items():
                if k in ("id", "prolific_id"):
                    continue
                o[k] = v
            o["demographics"] = dict(o["demographics"])
            if not args.keep_zip:
                o["demographics"].pop("zipcode", None)
            out.append(o)
        public[(pub_task, domain)] = out
        path = args.jsonl_out / f"{pub_task}_{domain}.jsonl"
        path.write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out))
        print(f"{path}: {len(out)} items, {len({o['participant_id'] for o in out})} participants")

    # Privacy lint over every string in the public files.
    problems = []
    for key, out in public.items():
        for o in out:
            blob = json.dumps(o, ensure_ascii=False)
            for name, rx in (("24-hex id", HEX24), ("email", EMAIL), ("phone", PHONE)):
                if rx.search(blob):
                    problems.append((key, o["item_id"], name))
            if not args.keep_zip and "zipcode" in o["demographics"]:
                problems.append((key, o["item_id"], "zipcode"))
    if problems:
        for p in problems[:20]:
            print("LINT", p)
        sys.exit(f"privacy lint failed: {len(problems)} hits")
    total = sum(len(v) for v in public.values())
    print(f"lint ok; {total} items, {len(pmap)} participants")

    if args.parquet_out:
        import pandas as pd
        args.parquet_out.mkdir(parents=True, exist_ok=True)
        for pub_task in set(TASKS.values()):
            frames = []
            for domain in DOMAINS:
                df = pd.DataFrame(public[(pub_task, domain)])
                for col in ("demographics", "context_qas", "answer_options", "source_qa", "scale"):
                    if col in df:
                        df[col] = df[col].map(lambda v: json.dumps(v, ensure_ascii=False))
                frames.append(df)
            df = pd.concat(frames, ignore_index=True)
            path = args.parquet_out / pub_task / "test-00000-of-00001.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, index=False)
            print(f"{path}: {len(df)} rows, {len(df.columns)} columns")


if __name__ == "__main__":
    main()
