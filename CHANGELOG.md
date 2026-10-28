# Changelog

Data versions are git tags in this repository and on the Hub dataset
(social-atoms/hugagent). Reported numbers cite the version they were computed on.

## data-v1.1 (2026-10-28)

Corrected 54 belief_dynamics_update items in the zoning domain, question 1.4r
(the "same upzoning rules across the whole city" scenario), one per participant.

Cause: the survey definition the item builder used listed reason code G for this
question, while the survey export stores that reason under code N. The builder
defaulted a missing code to 0, so each of these items carried `user_answer` 0,
outside the 1 to 5 scale, and named the wrong reason.

Fields changed on each item: `question_id` 1.4r_G to 1.4r_N, `reason_code` G to
N, `reason_text` to "Everyone should help handle the growth.", the reason phrase
inside `task_question`, and `user_answer` from 0 to the participant's recorded
rating. Item ids, participants, contexts and item counts are unchanged.

Builder fixes: `Benchmark/process_data.py` now raises on a reason code that is
missing from the export instead of writing 0, and
`Benchmark/survey_content/surveys.json` lists N for 1.4r.

Item ids (all `belief_dynamics_update-zoning-`): 0002, 0018, 0024, 0040, 0052, 0058, 0069, 0083, 0092, 0113, 0121, 0135, 0149, 0157, 0168, 0176, 0182, 0188, 0192, 0211, 0228, 0246, 0254, 0262, 0268, 0274, 0279, 0292, 0301, 0310, 0323, 0326, 0330, 0333, 0339, 0346, 0356, 0363, 0378, 0398, 0406, 0413, 0424, 0442, 0451, 0459, 0473, 0478, 0490, 0497, 0504, 0514, 0528, 0547.

Baselines in `results/baselines/` were re-scored on data-v1.1. The gpt-4o row
stays on data-v1.0 until it is re-run.

## data-v1.0 (2026-10-07)

First public release: 356 belief_state_inference and 1,386
belief_dynamics_update items, 54 participants, three domains.
