# Results

One JSON file per model and data version, written by `Benchmark/hugagent_eval.py --out`.
`scripts/make_leaderboard.py` turns this folder into the table in the main README and
`docs/leaderboard.json` for the project page.

Layout:

- `paper/` numbers copied from Table 2 of the paper (authors' pipeline, pre-release inputs).
- `baselines/` non-learning baselines computed on the released data by the scorer.
- `<model>/<data-version>.json` runs on the released data. Submit yours by pull request.

Required fields: `model`, `data_version`, `date`, `overall.bsi_acc`, `overall.bdu_acc`,
`overall.bdu_mae`, `overall.bdu_dir_acc`, `overall.ati`. Optional: `settings`, `per_domain`,
`source`, `notes`. A submission must come from an unmodified scorer run on the full set
(no `--limit`) and should link the raw responses (`--save-responses`) somewhere we can check.
