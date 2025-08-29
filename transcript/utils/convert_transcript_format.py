#!/usr/bin/env python3
"""
Convert merged transcripts to a simplified format with just prolific_id and transcript.
The transcript will be a list of question-answer pairs.
"""

import json
import os
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

def convert_qa_history_to_transcript(qa_history: Union[List[Any], Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Convert qa_history format to transcript format.
    
    Args:
        qa_history: QA history in various possible formats
        
    Returns:
        List of QA pairs in the new format with just question and answer fields
    """
    transcript = []
    
    # Debug logging
    logger.info(f"QA history type: {type(qa_history)}")
    if isinstance(qa_history, dict):
        logger.info(f"QA history keys: {list(qa_history.keys())}")
    
    # Handle dictionary format
    if isinstance(qa_history, dict):
        # If qa_history is a dict with numbered keys
        for key in sorted(qa_history.keys()):
            entry = qa_history[key]
            if isinstance(entry, dict):
                question = entry.get('question', '')
                answer = entry.get('answer', '')
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                question = entry[0]
                answer = entry[1]
            else:
                continue
                
            if question and answer:
                transcript.append({
                    'question': question,
                    'answer': answer
                })
    
    # Handle list format
    elif isinstance(qa_history, list):
        # If qa_history is a list of dictionaries
        if qa_history and isinstance(qa_history[0], dict):
            for qa in qa_history:
                question = qa.get('question', '')
                answer = qa.get('answer', '')
                if question and answer:
                    transcript.append({
                        'question': question,
                        'answer': answer
                    })
        # If qa_history is a list of alternating strings
        else:
            for i in range(0, len(qa_history), 2):
                if i + 1 < len(qa_history):
                    question = qa_history[i]
                    answer = qa_history[i + 1]
                    if question and answer:
                        transcript.append({
                            'question': question,
                            'answer': answer
                        })
    
    return transcript

def convert_file(input_file: str, output_dir: str = None) -> None:
    """
    Convert a single merged transcript file to the new format.
    
    Args:
        input_file: Path to the input JSON file
        output_dir: Optional directory to save the output file; if not provided,
                   saves in the same directory as input file
    """
    try:
        # Read input file
        logger.info(f"Processing file: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Debug logging
        logger.info(f"Input data type: {type(data)}")
        if isinstance(data, dict):
            logger.info(f"Input data keys: {list(data.keys())}")
        elif isinstance(data, list) and data:
            logger.info(f"First record type: {type(data[0])}")
            if isinstance(data[0], dict):
                logger.info(f"First record keys: {list(data[0].keys())}")
        
        # Process each record
        converted_records = []
        
        # Handle both list and dictionary input formats
        if isinstance(data, dict):
            data = [data]
        
        for record in data:
            agent_id = record.get('agent_id')
            qa_history = record.get('qa_history')
            
            if not agent_id or not qa_history:
                logger.warning(f"Skipping record - missing agent_id or qa_history. Keys: {list(record.keys())}")
                continue
            
            # Convert to new format
            converted_record = {
                'prolific_id': agent_id,
                'transcript': convert_qa_history_to_transcript(qa_history)
            }
            
            if converted_record['transcript']:  # Only add if we have valid transcript entries
                converted_records.append(converted_record)
        
        if not converted_records:
            logger.warning("No valid records were converted!")
            return
        
        # Determine output path
        input_path = Path(input_file)
        if output_dir:
            output_path = Path(output_dir) / f"{input_path.stem}_converted{input_path.suffix}"
        else:
            output_path = input_path.parent / f"{input_path.stem}_converted{input_path.suffix}"
        
        # Save converted records
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(converted_records, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Successfully converted {len(converted_records)} records to {output_path}")
        
    except Exception as e:
        logger.error(f"Error processing file {input_file}: {str(e)}")
        logger.debug("Full traceback:", exc_info=True)
        raise

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert merged transcripts to simplified format')
    parser.add_argument('-i', required=True, help='Path to the input merged transcript JSON file')
    parser.add_argument('-o', help='Optional directory to save the output file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    convert_file(args.i, args.o)

if __name__ == '__main__':
    main() 