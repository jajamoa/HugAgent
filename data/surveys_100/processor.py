#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Three-Topic Survey Processor   (updated 2025-05-15)

Read *.csv survey results → generate
  <stem>_demographics.json
  <stem>_housing_reactions.json
  <stem>_surveillance_reactions.json
  <stem>_healthcare_reactions.json

Output format satisfies Evaluator:
{
  "<PROLIFIC_ID>": {
    "opinions": { "1.1": 7, "1.2": 9, ... },
    "reasons":  { "1.1": { "A": 3, "B": 5 }, "1.2": {...} }
  },
  ...
}
"""
from __future__ import annotations

import re
import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any

import pandas as pd


class ThreeTopicSurveyProcessor:
    def __init__(
        self,
        input_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        schema_path: str | Path | None = None,
        housing_map: str | Path | None = None,
        camera_map: str | Path | None = None,
        health_map: str | Path | None = None,
        check_attention: bool = True,
    ):
        here = Path(os.path.dirname(os.path.abspath(__file__)))

        self.schema: Dict[str, Any] = self._load_json(
            schema_path or here / "surveys.json"
        )
        self.housing_mapping = self._load_json(
            housing_map or here / "housing_reason_mapping.json"
        )["mapping"]
        self.camera_mapping = self._load_json(
            camera_map or here / "surveillance_reason_mapping.json"
        )["mapping"]
        self.health_mapping = self._load_json(
            health_map or here / "healthcare_reason_mapping.json"
        )["mapping"]

        self.input_dir = Path(input_dir) if input_dir else here / "raw"
        self.output_dir = Path(output_dir) if output_dir else here / "processed"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.check_attention = check_attention

        self.demographic_columns = [
            "In the past five years, briefly describe your housing experience, including any moves, rental situations, and changes in your housing status. What were the reasons for these changes?",
            "What is your age?",
            "Have you moved in the past year?",
            "If you rent, what is your approximate monthly rent as a percentage of your income?",
            "What best describes your housing status?",
            "What is your primary mode of transportation? (Please select all that apply)",
            "What is your annual household income?",
            "Which of the following best describes your occupation?",
            "What's your marital status?",
            "Do you have any children under the age of 18 living with you?",
            "What is the age range of your children under 18 years old? (You may select more than one option.)",
            "What is your ZIP code?",
            "What is your race or ethnicity?",
            "Which of the following best describes your current financial situation?",
            "How safe do you feel in your neighborhood?",
            "Which of the following best describes your current health insurance and disability status?",
            "What is the highest level of education you have completed?",
            "What is your citizenship and nativity status?",
        ]

    @staticmethod
    def _load_json(path: Path | str) -> Dict[str, Any]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def process_csv_files(self) -> None:
        csv_files = list(self.input_dir.glob("*.csv"))
        if not csv_files:
            print(f"⚠️  No CSV files in {self.input_dir}")
            return
        for csv_file in csv_files:
            print(f"▶ Processing {csv_file.name}")
            self._process_single_file(csv_file)

    def _process_single_file(self, csv_path: Path) -> None:
        df = pd.read_csv(csv_path, dtype=object).fillna("").astype(str)
        df.columns = [c.replace("\n", " ").strip() for c in df.columns]

        id_col_candidates = [
            col for col in df.columns
            if re.search(r"\bprolific\s*id\b", col, flags=re.I)
        ]
        if not id_col_candidates:
            print(f"❌   No column containing 'Prolific ID' found in {csv_path.name}")
            return

        id_col = id_col_candidates[0]
        demographics, housing_all, camera_all, health_all = {}, {}, {}, {}

        # Define demographic field mappings with flexible matching patterns
        demographic_mappings = {
            "housing_experience": r"briefly describe your housing experience",
            "age": r"what is your age",
            "moved_last_year": r"have you moved in the past year",
            "rent_income_ratio": r"rent.*percentage.*income",
            "housing_status": r"describes your housing status",
            "transportation": r"mode of transportation",
            "household_income": r"annual household income",
            "occupation": r"describes your occupation",
            "marital_status": r"marital status",
            "has_children": r"children under.*18.*living with you",
            "children_age": r"age range of your children",
            "zipcode": r"zip code",
            "race_ethnicity": r"race or ethnicity",
            "financial_situation": r"financial situation",
            "neighborhood_safety": r"safe.*feel.*neighborhood",
            "health_insurance": r"health insurance.*disability status",
            "education": r"level of education",
            "citizenship": r"citizenship.*nativity status"
        }

        for pid in df[id_col].unique():
            part = df[df[id_col] == pid]

            if self.check_attention:
                attention_cols = [col for col in df.columns if "please press the following link" in col.lower()]
                attention_answers = [
                    part[col].iloc[0].strip().lower() for col in attention_cols if col in part.columns
                ]
                if not attention_answers or any(ans != "yes" for ans in attention_answers):
                    print(f"⏭️  Skipping {pid} due to failed attention check")
                    continue

            # Check for missing demographic data
            demo_data = {}

            has_missing_data = False
            for field, pattern in demographic_mappings.items():
                matching_cols = [col for col in df.columns if re.search(pattern, col.lower())]
                if matching_cols:
                    value = part[matching_cols[0]].iloc[0].strip()
                    if not value:  # Skip if any demographic field is empty
                        has_missing_data = True
                        break
                    demo_data[field] = value
                else:
                    has_missing_data = True
                    break

            if has_missing_data:
                print(f"⏭️  Skipping {pid} due to missing demographic data")
                continue

            demographics[pid] = demo_data

            # —— Topic reactions (prioritize by tags; keep original fallback logic) ——
            housing_all[pid] = self._extract_topic_reactions(
                part,
                self.schema["topics"]["upzoning"],
                self.housing_mapping,
                topic_tag="zoning",
            )

            camera_all[pid] = self._extract_topic_reactions(
                part,
                self.schema["topics"]["surveillance_camera"],
                self.camera_mapping,
                topic_tag="camera",
            )

            health_all[pid] = self._extract_topic_reactions(
                part,
                self.schema["topics"]["universal_healthcare"],
                self.health_mapping,
                topic_tag="healthcare",
            )

            # Check if any required opinions or reasons are missing
            if not all(len(reactions["opinions"]) > 0 and len(reactions["reasons"]) > 0 
                      for reactions in [housing_all[pid], camera_all[pid], health_all[pid]]):
                print(f"⏭️  Skipping {pid} due to missing opinions or reasons")
                del demographics[pid]
                if pid in housing_all: del housing_all[pid]
                if pid in camera_all: del camera_all[pid]
                if pid in health_all: del health_all[pid]
                continue

        stem = csv_path.stem
        (self.output_dir / f"{stem}_demographics.json").write_text(
            json.dumps(demographics, indent=2, ensure_ascii=False)
        )
        (self.output_dir / f"{stem}_housing_reactions.json").write_text(
            json.dumps(housing_all, indent=2, ensure_ascii=False)
        )
        (self.output_dir / f"{stem}_surveillance_reactions.json").write_text(
            json.dumps(camera_all, indent=2, ensure_ascii=False)
        )
        (self.output_dir / f"{stem}_healthcare_reactions.json").write_text(
            json.dumps(health_all, indent=2, ensure_ascii=False)
        )
        print(f"✅  Done: {csv_path.name}")

    @staticmethod
    def _safe_int(x: str) -> int | None:
        try:
            return int(float(x))
        except Exception:
            return None

    @staticmethod
    def _normalize(text: str) -> str:
        """
        - Remove all punctuation (including Chinese and English)
        - Unicode NFKD compatibility decomposition, remove accents
        - Compress whitespace, convert to lowercase
        """
        import unicodedata
        text = unicodedata.normalize("NFKD", text)
        text = re.sub(r"[^\w\s]", " ", text)          # Remove punctuation
        text = re.sub(r"\s+", " ", text).strip().lower()
        return text

    def _col_match(self, col: str, target: str) -> bool:
        """Match column name with target question text."""
        col = re.sub(r"\s+", " ", col.strip().lower())
        target = re.sub(r"\s+", " ", target.strip().lower())
        target_key = " ".join(target.split()[:7])
        if target_key and target_key in col:
            return True
        target_words = set(target.split())
        col_words = set(col.split())
        common_words = target_words & col_words
        if len(common_words) >= min(len(target_words) * 0.6, len(col_words) * 0.6):
            return True
        return False

    def _extract_topic_reactions(
        self,
        part_df: pd.DataFrame,
        topic_schema: Dict[str, Any],
        reason_map: Dict[str, str],
        topic_tag: str,   # ← New: used to match [zoning]/[camera]/[healthcare]
    ) -> Dict[str, Any]:
        out = {"opinions": {}, "reasons": {}}

        # Reverse word bag: used to match letter codes based on reason text
        reverse_map = {}
        for text, code in reason_map.items():
            clean_text = re.sub(r"[^\w\s]", "", text.lower())
            reverse_map.setdefault(code, set(clean_text.split()))

        # Pre-cache lowercase column names
        lower_cols = [(c, c.lower()) for c in part_df.columns]

        def _tagged_cols_for(q_id: str, kind: str) -> list[str]:
            """Find columns with tags: kind ∈ {'opinions','reasons'}"""
            tgt_topic = f"[{topic_tag}]"
            tgt_qid = f"[{q_id}]"
            tgt_kind = f"[{kind}]"
            return [
                c for c, cl in lower_cols
                if tgt_topic in cl and tgt_qid in cl and tgt_kind in cl
            ]

        for q in topic_schema["questions"]:
            q_id = q["id"]
            q_text = q["text"]

            matched_cols = [c for c in part_df.columns if self._col_match(c, q_text)]
            
            # ---------- Opinions (only non-reason_evaluation; prioritize tags, then fallback) ----------
            if q.get("type") != "reason_evaluation":
                opinion_cols = _tagged_cols_for(q_id, "opinions")
                opinion_val = None
                if opinion_cols:
                    raw = part_df[opinion_cols[0]].iloc[0]
                    opinion_val = self._safe_int(raw)
                else:
                    if matched_cols:
                        pref = next((c for c in matched_cols if "[opinions]" in c.lower()), matched_cols[0])
                        opinion_val = self._safe_int(part_df[pref].iloc[0])
                if opinion_val is not None:
                    out["opinions"][q_id] = opinion_val

            # ---------- Reasons (native reasons or followup) ----------
            def _collect_reasons_for(qid: str, needed_codes: list[str]) -> dict[str, int]:
                """Extract reasons for this question from tagged columns, filtered by needed_codes"""
                got: dict[str, int] = {}
                candidates = _tagged_cols_for(qid, "reasons")
                for col in candidates:
                    tokens = re.findall(r"\[([^\]]+)\]", col)
                    code_from_tag = None
                    if len(tokens) >= 4 and tokens[2].lower() == "reasons":
                        tcode = tokens[3].strip().upper()
                        if tcode in needed_codes:
                            code_from_tag = tcode
                    # Infer code from text (using the last bracketed reason text)
                    reason_txt = tokens[-1].strip() if tokens else ""
                    r_clean = re.sub(r"[^\w\s]", "", reason_txt.lower())
                    reason_words = set(r_clean.split())
                    best_code = None
                    best_overlap = 0.0
                    for code in needed_codes:
                        target_words = reverse_map.get(code, set())
                        if not target_words:
                            continue
                        overlap = len(reason_words & target_words) / max(1, len(target_words))
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_code = code
                    final_code = code_from_tag or best_code
                    if final_code and final_code in needed_codes:
                        val = self._safe_int(part_df[col].iloc[0])
                        if val is not None:
                            got[final_code] = val
                return got

            # 1) Current question is a reasons question
            if q.get("type") == "reason_evaluation":
                need = q["reasons"]
                out["reasons"].setdefault(q_id, {})
                # First extract from tags; if no tags, use your original loose matching (minimal changes)
                extracted = _collect_reasons_for(q_id, need)
                if not extracted:
                    # Fallback: In columns matching the question text, prioritize letter codes from tags; otherwise infer from word overlap with reason text
                    for col in part_df.columns:
                        if not self._col_match(col, q_text):
                            continue
                        tokens = re.findall(r"\[([^\]]+)\]", col)
                        # Directly read code from tags
                        code = next((t.strip().upper() for t in tokens if t.strip().upper() in need), None)
                        if not code:
                            # Infer from word overlap with the last bracketed text
                            reason_txt = tokens[-1].strip() if tokens else ""
                            r_clean = re.sub(r"[^\w\s]", "", reason_txt.lower())
                            reason_words = set(r_clean.split())
                            best_code, best_overlap = None, 0.0
                            for cand in need:
                                target_words = reverse_map.get(cand, set())
                                if not target_words:
                                    continue
                                overlap = len(reason_words & target_words) / max(1, len(target_words))
                                if overlap > best_overlap:
                                    best_overlap, best_code = overlap, cand
                        if code:
                            val = self._safe_int(part_df[col].iloc[0])
                            if val is not None:
                                extracted[code] = val
                out["reasons"][q_id].update(extracted)

            # 2) Current question has followup reasons (scenario question reasons)
            if q.get("has_reason_followup"):
                parent_id = q_id
                need = q["followup"]["reasons"]
                out["reasons"].setdefault(parent_id, {})
                extracted = _collect_reasons_for(parent_id, need)
                out["reasons"][parent_id].update(extracted)

        return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract demographics & reactions from three-topic survey CSVs"
    )
    parser.add_argument("--csv", help="Process single CSV file (path)")
    parser.add_argument("--in-dir", help="Directory with raw CSVs")
    parser.add_argument("--out-dir", help="Directory to save processed JSON")
    parser.add_argument("--no-attention-check", action="store_true", help="Disable attention check")
    args = parser.parse_args()

    proc = ThreeTopicSurveyProcessor(
        input_dir=args.in_dir,
        output_dir=args.out_dir,
        check_attention=not args.no_attention_check,
    )

    if args.csv:
        f = Path(args.csv)
        if f.exists() and f.suffix.lower() == ".csv":
            proc._process_single_file(f)
        else:
            print("❌  --csv must be an existing .csv file")
    else:
        proc.process_csv_files()
