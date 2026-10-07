# Data release rules

This file says what HugAgent publishes, what it keeps private, and how a release
is built. It applies to this repository and to the Hugging Face dataset.

## What is public

- The 1,742 benchmark items in `Benchmark/data/` (six jsonl files, one per task
  and domain) and the same items as parquet on the Hub.
- Survey wording and reason-code mappings in `Benchmark/survey_content/`.
- The evaluation code, prompts, and the synthetic track `Benchmark/synth_500.zip`.
- Model results we or others submit under `results/`.

## What stays private

- Raw interview transcripts, chatbot session logs, survey exports, and recruitment
  platform ids. They live in a private repository and are never committed here.
- The mapping from internal participant identifiers to public `participant_id`
  values (written by `scripts/build_release.py --map-out` outside the repo).
- Participant free text beyond what an item needs. An item carries only the
  interview answers it uses as context (`context_qas`).

The repository history was rewritten on 2026-10-07 to remove earlier commits
that contained raw participant folders. The branches `data_preprocess` and
`baseline-models` were deleted for the same reason. Do not restore them.

## Identifiers and demographics

- `participant_id` is `P01` to `P54`. It has no relation to any recruitment id.
  A 24-character hex string must never appear in a public file.
- `item_id` is `<task>-<domain>-<nnnn>` and is stable across releases. New items
  get new ids; an item is never renumbered or edited in place.
- The zipcode field is dropped. Age, income, race, occupation and household
  fields stay as collected because the tasks use them.
- `scripts/build_release.py` refuses to write a file that contains a 24-hex id,
  an email address, a phone number, or a zipcode field.

## Item construction and selection

Participants were filtered with the quality-control protocol in Appendix O of
the paper (redundant answers, meta-level questioning, insufficient length,
sparse causal networks): 54 of about 120 were retained.

Belief state inference items are built from the GT QAs of each transcript, the
short polarity judgments a participant gave during the interview. An item hides
one such judgment, shows the participant's other answers as context, and asks a
two-option question whose gold answer is the hidden judgment (`source_qa`).
Belief dynamics update items come from the questionnaire: baseline stance,
stance after each scenario, and the 1 to 5 reason weights. Questionnaire
answers never appear in the interview, so they cannot leak into the context.

Candidate items were produced by the build pipeline and then reviewed by hand
by the authors. The released set is exactly the set the paper reports on
(Table 1: 356 + 1,386). The internal file names carried suffixes from that
review stage; the public files are named by task and domain only.

## Versioning

- Releases are tagged `data-v1.0`, `data-v1.1`, and so on. The Hub dataset
  carries the same tag.
- A new version adds a CHANGELOG entry: items added, items withdrawn (with
  reason, never with content), schema changes.
- Reported numbers must cite the data version they were computed on.

## Reported scores

The test set is public with gold answers, so every leaderboard score is
self-reported. A submission must include the raw responses; we re-score them
and mark the row "released data". We do not verify that a model was not
trained on the data. Plan for v2: hold back a set of participants that is
never released (inputs and gold), score submissions on it ourselves, and show
those rows as "verified".

## Hugging Face Hub

- Repository `social-atoms/hugagent` (or the org that owns it), type dataset.
- Gated with automatic approval. Users accept the CC BY-NC 4.0 terms and the
  no-profiling clause before downloading; we keep the access log.
- Files: `data/<config>/test-00000-of-00001.parquet`, two configs,
  `belief_state_inference` and `belief_dynamics_update`, one `test` split each.
  No loading script. The card YAML declares the configs so `load_dataset` works
  without arguments beyond the config name.
- `hf/README.md` is the dataset card. It must state: collection method and
  dates, consent and IRB approval, the quality-control filter, the human
  test-retest ceiling, intended use, prohibited use, the canary string, and
  the citation.
- `hf/canary.txt` holds a canary string that also appears in the card. If the
  string shows up in a model's output, the model has seen the data.

## Release checklist

1. Run `scripts/build_release.py` on the internal files. The lint must pass and
   the counts must match Table 1 of the paper (356 BSI, 1,386 BDU, 1,742 total).
2. Load each parquet with `datasets` and spot-check ten random items by hand,
   including every string field, for names, places, or employer details.
3. Confirm the item-selection section above is complete.
4. Tag the commit, upload parquet and card to the Hub, enable gating, and test
   `load_dataset` from a fresh account.
5. Run the evaluation script on at least one model and commit the result JSON
   under `results/<model>/<data-version>.json`.

## Known issues to settle before data-v1.0

- Three belief state inference items (zoning 0030 to 0032, one participant)
  carry the gold answer `A/B`. The scorer accepts either letter for them.
- Belief state inference gold labels are 223 A versus 130 B (plus the three
  above). A constant "A" answer scores 62.6%. Report that baseline next to
  model numbers, or rebalance by swapping option order on half the items.
