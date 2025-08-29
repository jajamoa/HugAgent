import os
import json
import csv
import glob
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class TranscriptConfig:
    """Configuration for transcript processing."""
    input_dir: str  # Directory containing CSV files
    output_dir: str  # Directory for processed JSON files
    metadata_rows: int = 8  # Number of metadata rows before QA data
    id_field: str = "Prolific ID"  # Field name for participant ID in metadata
    question_column: str = "Question"  # Column name for questions in QA section
    answer_column: str = "Answer"  # Column name for answers in QA section
    min_answer_length: int = 2  # Minimum length for valid answers

class TranscriptProcessor:
    """Process interview transcripts from CSV to structured JSON format."""
    
    def __init__(self, config: TranscriptConfig):
        """Initialize the processor with configuration.
        
        Args:
            config: Configuration for transcript processing.
        """
        self.config = config
        self._ensure_output_dir()
        
    def _ensure_output_dir(self):
        """Ensure output directory exists."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
    def process_all_files(self) -> Dict[str, List[str]]:
        """Process all CSV files in the input directory.
        
        Returns:
            Dictionary with processing results:
            {
                'processed': [list of successfully processed files],
                'failed': [list of files that failed processing]
            }
        """
        results = {
            'processed': [],
            'failed': []
        }
        
        # Get all CSV files in input directory
        csv_files = glob.glob(os.path.join(self.config.input_dir, "*.csv"))
        
        for csv_file in csv_files:
            try:
                self.process_file(csv_file)
                results['processed'].append(csv_file)
                print(f"Successfully processed: {csv_file}")
            except Exception as e:
                print(f"ERROR processing {csv_file}: {str(e)}")
                results['failed'].append(csv_file)
                
        return results
    
    def _read_metadata(self, csv_path: str) -> Tuple[Dict[str, str], int]:
        """Read metadata from the CSV file header.
        
        Args:
            csv_path: Path to the CSV file.
            
        Returns:
            Tuple of (metadata dict, last metadata line number).
        """
        metadata = {}
        last_metadata_line = 0
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            # Process metadata section
            for i, line in enumerate(lines):
                line = line.strip()
                
                # Skip empty lines in metadata section
                if not line:
                    continue
                    
                # Stop if we've reached the maximum metadata rows
                if i >= self.config.metadata_rows:
                    break
                    
                try:
                    # Split only on first comma to handle values that contain commas
                    parts = line.split(',', 1)
                    if len(parts) == 2:
                        key, value = parts
                        key = key.strip()
                        value = value.strip().strip('"')
                        if key and value:  # Only store non-empty key-value pairs
                            metadata[key] = value
                            last_metadata_line = i
                except Exception as e:
                    print(f"Warning: Could not parse metadata line {i+1}: {line}")
                    continue
                    
        return metadata, last_metadata_line
    
    def _find_qa_start(self, lines: List[str], start_from: int) -> int:
        """Find the start of QA section, skipping empty lines and headers.
        
        Args:
            lines: List of file lines.
            start_from: Line number to start searching from.
            
        Returns:
            Line number where QA data starts.
        """
        for i in range(start_from, len(lines)):
            line = lines[i].strip()
            if line and "Question" in line and "Answer" in line:
                return i + 1  # Return the line after the header
        return start_from
    
    def process_file(self, csv_path: str):
        """Process a single CSV file into JSON transcripts."""
        with open(csv_path, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()

        # Read metadata using the existing method
        metadata, last_metadata_line = self._read_metadata(csv_path)
        participant_id = metadata.get(self.config.id_field)
        if not participant_id:
            raise ValueError(f"No {self.config.id_field} found in metadata")

        # Identify QA header line
        qa_header_idx = None
        for idx, line in enumerate(all_lines):
            if "Question Number" in line and self.config.question_column in line and self.config.answer_column in line:
                qa_header_idx = idx
                break

        if qa_header_idx is None:
            raise ValueError(f"QA header with '{self.config.question_column}' and '{self.config.answer_column}' not found in {csv_path}")

        # Read QA part from the header line
        qa_reader = csv.DictReader(all_lines[qa_header_idx:])

        qa_pairs = []
        for row in qa_reader:
            question = row.get(self.config.question_column, '').strip()
            answer = row.get(self.config.answer_column, '').strip()

            if (question and answer and len(answer) >= self.config.min_answer_length
                and not question.lower().startswith("question number")):
                qa_pairs.append({
                    "question": question,
                    "answer": answer
                })

        # Create final transcript
        transcript_data = {
            "prolific_id": participant_id,
            "metadata": metadata,
            "transcript": qa_pairs
        }

        output_path = os.path.join(self.config.output_dir, f"{participant_id}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(transcript_data, f, indent=2, ensure_ascii=False)

        print(f"Saved transcript for {participant_id} with {len(qa_pairs)} QA pairs to: {output_path}")
        
def process_transcripts(
    input_dir: str,
    output_dir: str,
    metadata_rows: int = 8,
    id_field: str = "Prolific ID",
    question_column: str = "Question",
    answer_column: str = "Answer",
    min_answer_length: int = 2
) -> Dict[str, List[str]]:
    """Convenience function to process all transcripts in a directory.
    
    Args:
        input_dir: Directory containing CSV files.
        output_dir: Directory for processed JSON files.
        metadata_rows: Number of metadata rows before QA data.
        id_field: Field name for participant ID in metadata.
        question_column: Column name for questions in QA section.
        answer_column: Column name for answers in QA section.
        min_answer_length: Minimum length for valid answers.
        
    Returns:
        Dictionary with processing results.
    """
    config = TranscriptConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        metadata_rows=metadata_rows,
        id_field=id_field,
        question_column=question_column,
        answer_column=answer_column,
        min_answer_length=min_answer_length
    )
    
    processor = TranscriptProcessor(config)
    return processor.process_all_files()

# Example usage:
if __name__ == "__main__":
    # Example directory structure
    base_dir = os.path.dirname(os.path.dirname(__file__))
    input_dir = os.path.join(base_dir, "data", "raw_transcripts")
    output_dir = os.path.join(base_dir, "data", "processed_transcript")
    
    # Process all transcripts
    results = process_transcripts(
        input_dir=input_dir,
        output_dir=output_dir
    )
    
    print(f"\nProcessing Summary:")
    print(f"Successfully processed: {len(results['processed'])} files")
    print(f"Failed to process: {len(results['failed'])} files")
    
    if results['failed']:
        print("\nFailed files:")
        for file in results['failed']:
            print(f"- {file}")
