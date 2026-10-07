#!/usr/bin/env python3
"""Verify that the train split shares nothing with the test split.

Checks participant ids, exact items (task question + full context), and any
context QA answer text. Exit code 1 on any overlap. The release has no train
split; build one from the synthetic track with scripts/build_train_split.py
into a directory outside hf/ and point --train-dir at it.

  python scripts/build_train_split.py --parquet-out /tmp/hugagent-train --odyssim-out /tmp/hugagent-train/odyssim
  python scripts/check_disjoint.py --train-dir /tmp/hugagent-train
"""
import argparse, hashlib, json, sys
from pathlib import Path
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent


def rows(path):
    return pq.read_table(path).to_pylist()


def h(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", type=Path, default=ROOT / "hf/data")
    ap.add_argument("--train-dir", type=Path, required=True)
    args = ap.parse_args()
    bad = 0
    for cfg in ("belief_state_inference", "belief_dynamics_update"):
        test = rows(args.test_dir / cfg / "test-00000-of-00001.parquet")
        train = rows(args.train_dir / cfg / "train-00000-of-00001.parquet")
        t_p = {r["participant_id"] for r in test}
        n_p = {r["participant_id"] for r in train}
        t_items = {h(r["task_question"] + "\n" + r["context_qas"]) for r in test}
        n_items = {h(r["task_question"] + "\n" + r["context_qas"]) for r in train}
        t_ans = {h(qa["answer"].strip()) for r in test for qa in json.loads(r["context_qas"]) if qa.get("answer", "").strip()}
        n_ans = {h(qa["answer"].strip()) for r in train for qa in json.loads(r["context_qas"]) if qa.get("answer", "").strip()}
        splits = {r.get("split") for r in test} | {r.get("split") for r in train}
        print(f"{cfg}: test {len(test)} rows / {len(t_p)} participants, train {len(train)} rows / {len(n_p)} participants")
        print(f"  shared participants: {len(t_p & n_p)}  shared items: {len(t_items & n_items)}  shared context answers: {len(t_ans & n_ans)}  split labels: {sorted(splits)}")
        bad += len(t_p & n_p) + len(t_items & n_items) + len(t_ans & n_ans)
        bad += sum(r.get("split") != "test" for r in test) + sum(r.get("split") != "train" for r in train)
    print("OK: train and test are disjoint" if bad == 0 else f"FAIL: {bad} overlaps")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
