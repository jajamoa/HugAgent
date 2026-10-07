#!/usr/bin/env python3
"""Rebuild the leaderboard table in README.md and docs/leaderboard.json from results/."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
START, END = "<!-- leaderboard:start -->", "<!-- leaderboard:end -->"


def main():
    rows = []
    for p in sorted((ROOT / "results").rglob("*.json")):
        r = json.loads(p.read_text())
        o = r.get("overall", {})
        if "ati" not in o:
            continue
        rows.append({"model": r["model"], "data_version": r.get("data_version", ""), "date": r.get("date", ""),
                     "source": "paper" if p.parent.name == "paper" else ("baseline" if p.parent.name == "baselines" else "released data"),
                     "bsi_acc": o["bsi_acc"], "bdu_acc": o["bdu_acc"], "bdu_mae": o["bdu_mae"],
                     "bdu_dir_acc": o["bdu_dir_acc"], "ati": o["ati"], "file": str(p.relative_to(ROOT))})
    rows.sort(key=lambda r: -r["ati"])
    (ROOT / "docs" / "leaderboard.json").write_text(json.dumps({"rows": rows}, indent=1) + "\n")

    lines = ["| Model | BSI acc | BDU acc | BDU MAE | BDU dir. acc | ATI | Data | Source |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['bsi_acc']:.2f} | {r['bdu_acc']:.2f} | {r['bdu_mae']:.2f} | "
                     f"{r['bdu_dir_acc']:.2f} | {r['ati']:.2f} | {r['data_version']} | {r['source']} |")
    table = "\n".join(lines)
    readme = ROOT / "README.md"
    s = readme.read_text()
    a, b = s.index(START) + len(START), s.index(END)
    readme.write_text(s[:a] + "\n" + table + "\n" + s[b:])
    print(f"{len(rows)} rows")


if __name__ == "__main__":
    main()
