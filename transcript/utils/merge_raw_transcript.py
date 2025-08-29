#!/usr/bin/env python3
"""
Combine raw transcripts.

This script is used to merge all JSON files in the raw_transcripts directory,
only keep the first occurrence of each prolificId, and only keep the agent_id (using prolificId)
and qa_history fields, other information is ignored.

This script supports large JSON file processing, using stream parsing to avoid memory overflow.
"""

import json
import os
import glob
import argparse
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, List, Set, Generator, Optional
import time
import logging
import ijson  # used for stream processing large JSON files
from decimal import Decimal

# set log
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

# custom JSON encoder, handle special types like Decimal
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)

def safe_json_dumps(obj, **kwargs):
    """safe JSON serialization, handle special types"""
    return json.dumps(obj, cls=CustomJSONEncoder, **kwargs)

def process_file_stream(file_path: str, processed_ids: Set[str]) -> Generator[Dict[str, Any], None, None]:
    """
        Use stream processing to handle large JSON files and generate simplified records.
    
    Args:
        file_path: JSON file path
        processed_ids: processed prolificId set
        
    Yields:
        simplified record, only contains agent_id and qa_history
    """
    file_name = os.path.basename(file_path)
    logger.info(f"stream processing file: {file_name}")
    
    # counter
    added_count = 0
    skipped_count = 0
    error_count = 0
    processed_count = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # use ijson to stream parse each item in the array
            parser = ijson.items(f, 'item')
            
            for item in parser:
                processed_count += 1
                if processed_count % 100 == 0:
                    logger.debug(f"processed {processed_count} records")
                
                try:
                    # get prolificId
                    prolific_id = item.get("prolificId")
                    
                    if not prolific_id:
                        error_count += 1
                        continue
                    
                    # if already processed this prolificId, skip
                    if prolific_id in processed_ids:
                        skipped_count += 1
                        continue
                    
                    # find graphData containing qa_history
                    qa_history = None
                    for graph in item.get("graphs", []):
                        graph_data = graph.get("graphData", {})
                        if "qa_history" in graph_data:
                            qa_history = graph_data["qa_history"]
                            break
                    
                    if not qa_history:
                        error_count += 1
                        continue
                    
                    # mark as processed
                    processed_ids.add(prolific_id)
                    
                    # create simplified record
                    simplified_record = {
                        "agent_id": prolific_id,
                        "qa_history": qa_history
                    }
                    
                    # return simplified record
                    added_count += 1
                    yield simplified_record
                    
                except Exception as e:
                    error_count += 1
                    logger.warning(f"error when processing record: {str(e)}")
                    continue
        
        logger.info(f"added {added_count} new agents from {file_name}, skipped {skipped_count} duplicate agents, error {error_count} records, processed {processed_count} records")
        
    except Exception as e:
        logger.error(f"error when processing file {file_name}: {str(e)}")
        logger.debug(traceback.format_exc())

def process_file(file_path: str, processed_ids: Set[str]) -> Generator[Dict[str, Any], None, None]:
    """
    Process a single JSON file and generate simplified records.
    
    If the file size is larger than 100MB, use stream processing; otherwise, use standard processing.
    
    Args:
        file_path: JSON file path
        processed_ids: processed prolificId set
        
    Yields:
        simplified record, only contains agent_id and qa_history
    """
    # check file size
    file_size = os.path.getsize(file_path)
    if file_size > 100 * 1024 * 1024:  # if file size is larger than 100MB
        logger.info(f"file size: {file_size/1024/1024:.2f}MB, using stream processing")
        yield from process_file_stream(file_path, processed_ids)
        return
    
    file_name = os.path.basename(file_path)
    logger.info(f"processing file: {file_name} ({file_size/1024/1024:.2f}MB)")
    
    # counter
    added_count = 0
    skipped_count = 0
    error_count = 0
    
    try:
        # read JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # process each record
        for item in data:
            try:
                # get prolificId
                prolific_id = item.get("prolificId")
                
                if not prolific_id:
                    error_count += 1
                    continue
                
                # if already processed this prolificId, skip
                if prolific_id in processed_ids:
                    skipped_count += 1
                    continue
                
                # find graphData containing qa_history
                qa_history = None
                for graph in item.get("graphs", []):
                    graph_data = graph.get("graphData", {})
                    if "qa_history" in graph_data:
                        qa_history = graph_data["qa_history"]
                        break
                
                if not qa_history:
                    error_count += 1
                    continue
                
                # mark as processed
                processed_ids.add(prolific_id)
                
                # create simplified record
                simplified_record = {
                    "agent_id": prolific_id,
                    "qa_history": qa_history
                }
                
                # return simplified record
                added_count += 1
                yield simplified_record
                
            except Exception as e:
                error_count += 1
                logger.warning(f"error when processing record: {str(e)}")
                continue
        
        logger.info(f"added {added_count} new agents from {file_name}, skipped {skipped_count} duplicate agents, error {error_count} records")
        
    except Exception as e:
        logger.error(f"error when processing file {file_name}: {str(e)}")
        logger.debug(traceback.format_exc())

def save_json_in_chunks(data: List[Dict[str, Any]], output_file: Path, chunk_size: int = 1000) -> None:
    """
    Save large JSON data in chunks to avoid writing too much data at once.
    
    Args:
        data: data to save
        output_file: output file path
        chunk_size: size of each write chunk
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        # write starting bracket
        f.write('[\n')
        
        # write data in chunks
        for i, item in enumerate(data):
            json_str = safe_json_dumps(item, ensure_ascii=False)
            
            if i < len(data) - 1:
                f.write(json_str + ',\n')
            else:
                f.write(json_str + '\n')
            
            # flush buffer after writing chunk_size items
            if (i + 1) % chunk_size == 0:
                f.flush()
        
        # write ending bracket
        f.write(']\n')

def merge_transcripts(input_dir: str = None, output_dir: str = None):
    """
    Merge all JSON files in the raw_transcripts directory.
    
    Process:
    1. read all transcript files
    2. merge by prolificId, keep the first occurrence
    3. only keep agent_id and qa_history fields
    4. save merged data to a single JSON file
    
    Args:
        input_dir: optional, input directory path; if not provided, use default raw_transcripts directory
        output_dir: optional, output directory path; if not provided, use default processed_transcript directory
    """
    # get script directory
    script_dir = Path(__file__).parent.resolve()
    
    # set input and output directories
    if input_dir:
        raw_dir = Path(input_dir)
    else:
        raw_dir = script_dir.parent / "data" / "raw_transcripts"
    
    if output_dir:
        out_dir = Path(output_dir)
    else:
        out_dir = script_dir.parent / "data" / "processed_transcript"
    
    os.makedirs(out_dir, exist_ok=True)
    
    # find all JSON files
    json_files = sorted(glob.glob(str(raw_dir / "*.json")))
    
    if not json_files:
        logger.warning(f"no JSON files found in {raw_dir}")
        return
    
    logger.info(f"found {len(json_files)} JSON files to merge")
    
    # set for storing processed prolificId
    processed_ids = set()
    
    # final merged data
    merged_data = []
    
    # process each file
    start_time = time.time()
    for file_path in json_files:
        try:
            # process file and only keep new records
            for record in process_file(file_path, processed_ids):
                merged_data.append(record)
        except Exception as e:
            logger.error(f"error when processing file: {str(e)}")
            logger.debug(traceback.format_exc())
    
    # generate output file name
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = out_dir / f"merged_transcript_{timestamp}.json"
    fixed_output_file = out_dir / "merged_transcript.json"
    
    # save merged data
    try:
        # choose storage method based on data size
        if len(merged_data) > 5000:
            logger.info(f"large data size ({len(merged_data)} records), using chunked write")
            save_json_in_chunks(merged_data, output_file)
            save_json_in_chunks(merged_data, fixed_output_file)
        else:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(merged_data, f, ensure_ascii=False, indent=2, cls=CustomJSONEncoder)
            
            # create a fixed name copy, for frequent calls
            with open(fixed_output_file, 'w', encoding='utf-8') as f:
                json.dump(merged_data, f, ensure_ascii=False, indent=2, cls=CustomJSONEncoder)
        
        logger.info(f"\nmerge completed!")
        logger.info(f"total {len(merged_data)} unique agents")
        logger.info(f"time: {time.time() - start_time:.2f} seconds")
        logger.info(f"timestamp output file: {output_file}")
        logger.info(f"fixed name output file: {fixed_output_file}")
        
    except Exception as e:
        logger.error(f"error when saving output file: {str(e)}")
        logger.debug(traceback.format_exc())

def check_dependencies() -> bool:
    """check if dependencies are installed"""
    try:
        import ijson
        return True
    except ImportError:
        logger.error("missing required dependencies: ijson")
        logger.error("please install dependencies: pip install ijson")
        return False

def parse_args():
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description='merge transcript files')
    parser.add_argument('-i', '--input', help='input directory path')
    parser.add_argument('-o', '--output', help='output directory path')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # set verbose level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # check dependencies
    if not check_dependencies():
        logger.error("missing required dependencies, exit")
        sys.exit(1)
    
    # execute merge
    merge_transcripts(args.input, args.output)
