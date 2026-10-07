#!/usr/bin/env python3
"""HugAgent scorer. Works with any OpenAI-compatible chat endpoint.

  python Benchmark/hugagent_eval.py --model gpt-4o --base-url https://api.openai.com/v1 \
      --api-key-env OPENAI_API_KEY --limit 20          # smoke test, 20 items per file
  python Benchmark/hugagent_eval.py --model gpt-4o --api-key-env OPENAI_API_KEY \
      --out results/gpt-4o/data-v1.0.json               # full run, 1,742 items
  python Benchmark/hugagent_eval.py --baseline majority --out results/baselines/majority.json

Prompts are the ones from the paper (same text as evaluate_qwen.py). Metrics follow
Appendix R.1: BSI accuracy; BDU tolerance accuracy, MAE on a 5-point scale,
directional accuracy (lambda = 0.3), and the ATI score rescaled so that the paper's
random-guess baseline is 0 and the human test-retest ceiling is 100.
"""
import argparse, json, os, random, re, sys, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOMAINS = ["healthcare", "surveillance", "zoning"]
BASELINE_Q = {"healthcare": "3.1", "surveillance": "2.1", "zoning": "1.1"}
LAMBDA = 0.3
MAE_MAX = 3.0  # not stated in the paper; 3.0 reproduces the paper's GPT-4o ATI to within 0.5
# Paper Table 2, averaged over domains. These fix the 0 and 100 points of the ATI scale.
RANDOM = {"bsi_acc": 51.89, "bdu_acc": 43.12, "bdu_mae": 1.88, "bdu_dir_acc": 46.74}
HUMAN = {"bsi_acc": 84.84, "bdu_acc": 85.66, "bdu_mae": 0.68, "bdu_dir_acc": 88.92}

BSI_SYSTEM = ("You are an expert psychologist specializing in Theory of Mind and belief attribution. "
    "Your task: analyze conversation transcripts to infer what the participant believes about causal relationships. "
    "Focus on understanding their mental model - what they think causes what, not what is objectively true. "
    "Consider their background, conversation patterns, and implicit beliefs expressed through their responses. "
    "Base your inference strictly on evidence from their statements, not general assumptions.")
BDU_SYSTEM = ("You are an expert in survey research and human psychology. "
    "Your task: predict how this person would respond to a specific survey question based on their background and conversation. "
    "Consider their demographics, expressed opinions, and conversation patterns. "
    "Focus on understanding their likely response pattern, not what would be objectively correct. "
    "Base your prediction on evidence from their profile and statements.")


def load_items(data_dir: Path, limit=None):
    items = []
    for task in ("belief_state_inference", "belief_dynamics_update"):
        for domain in DOMAINS:
            rows = [json.loads(l) for l in (data_dir / f"{task}_{domain}.jsonl").read_text().splitlines() if l.strip()]
            items.extend(rows[:limit] if limit else rows)
    return items


def demo_block(d):
    return "Person's Background:\n" + "\n".join(f"- {k.replace('_', ' ').title()}: {v}" for k, v in d.items())


def context_block(qas, title):
    return title + "\n" + "\n\n".join(f"Q{i}: {qa['question']}\nA{i}: {qa['answer']}" for i, qa in enumerate(qas, 1))


def build_prompt(it, use_demo=True, use_context=True):
    parts = []
    if use_demo and it.get("demographics"):
        parts.append(demo_block(it["demographics"]))
    if it["task_type"] == "belief_attribution":
        if use_context and it.get("context_qas"):
            parts.append(context_block(it["context_qas"], "Conversation History:"))
        parts.append(f"Task: {it['task_question']}")
        keys = list(it["answer_options"].keys())
        parts.append("Answer options:\n" + "\n".join(f"{k}) {it['answer_options'][k]}" for k in keys))
        parts.append(f"Based on the evidence above, respond with ONLY the single letter ({', '.join(keys)}) that best represents this person's belief.")
        return BSI_SYSTEM, "\n\n".join(parts)
    if use_context and it.get("context_qas"):
        parts.append(context_block(it["context_qas"], "Previous Conversation:"))
    lo, hi = it.get("scale", [1, 10])
    if it.get("question_type") == "opinion":
        parts += [f"Survey Question: {it['task_question']}", f"Scale: {lo} to {hi}",
                  "Based on this person's profile and conversation, what number would they likely choose? Respond with ONLY the number."]
    else:
        parts += [f"Survey Question: {it['task_question']}", f"Context: This asks about the influence of: {it.get('reason_text', '')}",
                  f"Scale: {lo} to {hi} (1=no influence, {hi}=very strong influence)",
                  "Based on this person's profile and conversation, what rating would they likely give? Respond with ONLY the number."]
    return BDU_SYSTEM, "\n\n".join(parts)


def extract(it, text):
    if text is None:
        return None
    if it["task_type"] == "belief_attribution":
        keys = [k.upper() for k in it["answer_options"]]
        up = text.upper()
        found = [k for k in keys if k in up]
        if len(found) == 1:
            return found[0]
        return next((c for c in up if c in keys), None)
    nums = re.findall(r"\b\d+\b", text.strip())
    return int(nums[0]) if nums else None


def score(items, preds):
    """preds: item_id -> extracted answer (letter, int, or None). Returns the metrics dict."""
    by_domain = {d: {"bsi": [], "bdu": []} for d in DOMAINS}
    for it in items:
        by_domain[it["topic"]]["bsi" if it["task_type"] == "belief_attribution" else "bdu"].append(it)
    per_domain = {}
    for d in DOMAINS:
        bsi, bdu = by_domain[d]["bsi"], by_domain[d]["bdu"]
        m = {"n_bsi": len(bsi), "n_bdu": len(bdu)}
        if bsi:
            ok = [preds.get(it["item_id"]) is not None and str(preds[it["item_id"]]).upper()
                  in [a.strip().upper() for a in str(it["answer"]).split("/")] for it in bsi]
            m["bsi_acc"] = 100 * sum(ok) / len(ok)
            m["bsi_unparsed"] = sum(preds.get(it["item_id"]) is None for it in bsi)
        if bdu:
            errs, hits = [], []
            for it in bdu:
                p = preds.get(it["item_id"])
                lo, hi = it.get("scale", [1, 10])
                tol = 1 if hi - lo <= 5 else 2
                if p is None:
                    hits.append(False); errs.append(MAE_MAX * 2)  # unparsed counts as a large error
                    continue
                e = abs(p - it["user_answer"])
                hits.append(e <= tol)
                errs.append(e / 2 if hi - lo > 5 else e)  # 10-point errors mapped to the 5-point scale
            m["bdu_acc"] = 100 * sum(hits) / len(hits)
            m["bdu_mae"] = sum(errs) / len(errs)
            m["bdu_unparsed"] = sum(preds.get(it["item_id"]) is None for it in bdu)
            # Directional accuracy: scenario stance items paired with the same participant's baseline stance item.
            base = {it["participant_id"]: it for it in bdu
                    if it.get("question_type") == "opinion" and it["question_id"] == BASELINE_Q[d]}
            det, dirs = [], []
            for it in bdu:
                if it.get("question_type") != "opinion" or it["question_id"] == BASELINE_Q[d]:
                    continue
                b = base.get(it["participant_id"])
                if b is None or preds.get(it["item_id"]) is None or preds.get(b["item_id"]) is None:
                    continue
                dy = it["user_answer"] - b["user_answer"]
                dp = preds[it["item_id"]] - preds[b["item_id"]]
                det.append((dy == 0) == (dp == 0))
                if dy != 0 and dp != 0:
                    dirs.append((dy > 0) == (dp > 0))
            m["bdu_dir_pairs"] = len(det)
            if det:
                m["bdu_dir_acc"] = 100 * (LAMBDA * sum(det) / len(det) + (1 - LAMBDA) * (sum(dirs) / len(dirs) if dirs else 0))
        per_domain[d] = m
    overall = {}
    for k in ("bsi_acc", "bdu_acc", "bdu_mae", "bdu_dir_acc"):
        vals = [per_domain[d][k] for d in DOMAINS if k in per_domain[d]]
        if vals:
            overall[k] = sum(vals) / len(vals)
    overall.update({"n_items": len(items), "unparsed": sum(v is None for v in preds.values())})
    if all(k in overall for k in ("bsi_acc", "bdu_acc", "bdu_mae", "bdu_dir_acc")):
        u = ati_unscaled(overall)
        overall["ati_unscaled"] = u
        overall["ati"] = 100 * (u - ati_unscaled(RANDOM)) / (ati_unscaled(HUMAN) - ati_unscaled(RANDOM))
    return {"overall": overall, "per_domain": per_domain}


def ati_unscaled(m):
    s_mae = max(0.0, min(1.0, 1 - m["bdu_mae"] / MAE_MAX))
    return 0.5 * m["bsi_acc"] / 100 + 0.5 * (0.5 * (0.5 * s_mae + 0.5 * m["bdu_acc"] / 100) + 0.5 * m["bdu_dir_acc"] / 100)


def baseline_preds(items, kind, seed=0):
    rng = random.Random(seed)
    preds = {}
    for it in items:
        if it["task_type"] == "belief_attribution":
            preds[it["item_id"]] = "A" if kind == "majority" else rng.choice(list(it["answer_options"]))
        else:
            lo, hi = it.get("scale", [1, 10])
            # Global majority as in the paper: the single most frequent answer over all BDU items (1).
            preds[it["item_id"]] = 1 if kind == "majority" else rng.randint(lo, hi)
    return preds


def call_model(client, model, system, user, temperature, max_retries=3):
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(model=model, temperature=temperature,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            return r.choices[0].message.content
        except Exception as e:  # noqa: BLE001
            if attempt == max_retries - 1:
                sys.stderr.write(f"giving up on one item: {e}\n")
                return None
            time.sleep(2 ** attempt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--baseline", choices=["majority", "random"],
                    help="non-learning baseline, no API calls: majority = most frequent answer overall (A, 1), random = uniform")
    ap.add_argument("--data-dir", type=Path, default=HERE / "data")
    ap.add_argument("--data-version", default="data-v1.0")
    ap.add_argument("--limit", type=int, help="items per file, for smoke tests")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--no-demographics", action="store_true")
    ap.add_argument("--no-context", action="store_true")
    ap.add_argument("--seeds", type=int, default=5, help="random baseline: number of seeds to average")
    ap.add_argument("--out", type=Path, help="write the results JSON here")
    ap.add_argument("--save-responses", type=Path, help="also dump raw model responses (jsonl)")
    args = ap.parse_args()
    if not args.model and not args.baseline:
        ap.error("--model or --baseline is required")

    items = load_items(args.data_dir, args.limit)
    if args.baseline == "random":
        # Average the metrics over several seeds, as the paper does (mean of 5 runs).
        runs = [score(items, baseline_preds(items, "random", seed=s)) for s in range(args.seeds)]
        res = {"overall": {}, "per_domain": {}}
        for k in runs[0]["overall"]:
            res["overall"][k] = sum(r["overall"][k] for r in runs) / len(runs)
        for d in DOMAINS:
            res["per_domain"][d] = {k: sum(r["per_domain"][d][k] for r in runs) / len(runs) for k in runs[0]["per_domain"][d]}
        name = f"Random guess (mean of {args.seeds} seeds)"
        preds = None
    elif args.baseline:
        preds = baseline_preds(items, args.baseline)
        name = "Global majority (A, 1)"
    else:
        from openai import OpenAI
        key = os.environ.get(args.api_key_env)
        if not key:
            sys.exit(f"set {args.api_key_env} (or pass --api-key-env)")
        client = OpenAI(api_key=key, base_url=args.base_url)
        name = args.model
        preds, raw = {}, {}

        def one(it):
            system, user = build_prompt(it, not args.no_demographics, not args.no_context)
            text = call_model(client, args.model, system, user, args.temperature)
            return it["item_id"], text

        with ThreadPoolExecutor(args.workers) as ex:
            futs = [ex.submit(one, it) for it in items]
            for i, f in enumerate(as_completed(futs), 1):
                iid, text = f.result()
                raw[iid] = text
                if i % 100 == 0 or i == len(items):
                    sys.stderr.write(f"\r{i}/{len(items)}")
            sys.stderr.write("\n")
        by_id = {it["item_id"]: it for it in items}
        preds = {iid: extract(by_id[iid], t) for iid, t in raw.items()}
        if args.save_responses:
            args.save_responses.parent.mkdir(parents=True, exist_ok=True)
            with open(args.save_responses, "w") as f:
                for iid, t in raw.items():
                    f.write(json.dumps({"item_id": iid, "response": t, "parsed": preds[iid]}) + "\n")

    if preds is not None:
        res = score(items, preds)
    report = {
        "model": name, "data_version": args.data_version,
        "settings": {"temperature": args.temperature, "demographics": not args.no_demographics,
                     "context": not args.no_context, "limit": args.limit, "base_url": args.base_url,
                     "mae_max": MAE_MAX, "lambda": LAMBDA},
        "date": time.strftime("%Y-%m-%d"),
        **res,
    }
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        print(f"wrote {args.out}")
    o = res["overall"]
    print(f"{name}: BSI acc {o.get('bsi_acc', float('nan')):.2f}  BDU acc {o.get('bdu_acc', float('nan')):.2f}  "
          f"MAE {o.get('bdu_mae', float('nan')):.2f}  dir {o.get('bdu_dir_acc', float('nan')):.2f}  ATI {o.get('ati', float('nan')):.2f}")


if __name__ == "__main__":
    main()
