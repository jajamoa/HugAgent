import json
import os
from datetime import datetime
from typing import List, Dict, Any

def extract_research_transcripts(input_json_path: str, output_dir: str) -> None:
    """Extracts research Q&A pairs and saves them in structured JSON files by Prolific ID.

    Args:
        input_json_path: Path to the JSON file containing session data.
        output_dir: Directory to save processed transcript JSON files.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(input_json_path, "r", encoding="utf-8") as f:
        sessions = json.load(f)

    for session in sessions:
        prolific_id = session.get("prolificId")
        session_id = session.get("id")
        status = session.get("status")
        created_at = session.get("createdAt")
        updated_at = session.get("updatedAt")
        completed_at = session.get("completedAt", None)
        progress = session.get("progress", {})
        current = progress.get("current", 0)
        total = progress.get("total", 0)

        if not prolific_id:
            print(f"Skipping session without prolificId: {session_id}")
            continue

        metadata = {
            "Session ID": session_id,
            "Prolific ID": prolific_id,
            "Status": status,
            "Progress": f"{current}/{total}",
            "Created At": _format_datetime(created_at),
            "Updated At": _format_datetime(updated_at),
        }

        if completed_at:
            metadata["Completed At"] = _format_datetime(completed_at)

        transcript = [
            {"question": qa["question"], "answer": qa["answer"]}
            for qa in session.get("qaPairs", [])
            if qa.get("category") != "tutorial" and qa.get("answer", "").strip()
        ]

        output_data = {
            "prolific_id": prolific_id,
            "metadata": metadata,
            "transcript": transcript
        }

        output_file = os.path.join(output_dir, f"{prolific_id}.json")
        with open(output_file, "w", encoding="utf-8") as out_f:
            json.dump(output_data, out_f, indent=2, ensure_ascii=False)

        print(f"Saved transcript for {prolific_id} to {output_file}")

def _format_datetime(iso_date: str) -> str:
    """Convert ISO datetime string to human-readable format."""
    try:
        dt = datetime.fromisoformat(iso_date.replace("Z", ""))
        return dt.strftime("%-m/%-d/%Y, %-I:%M:%S %p")
    except Exception:
        return iso_date or ""

# Example usage
if __name__ == "__main__":
    input_json = "src/evaluation/models/m06_transcript/data/raw_transcripts/all-sessions-qa-data-2025-05-15T16-02-07-496Z.json"  # Replace with your actual input file
    output_directory = "src/evaluation/models/m06_transcript/data/processed_transcript"
    extract_research_transcripts(input_json, output_directory)
