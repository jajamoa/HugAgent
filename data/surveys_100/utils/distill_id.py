import json
import csv
import os

def extract_ids():
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define input and output file paths
    json_file = os.path.join(current_dir, 'processed/39ppl_ext_Survey on Upzoning ｜ Surveillance camera ｜ Universal Healthcare_demographics.json')
    output_file = os.path.join(current_dir, 'survey_ids.csv')
    
    # Read JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract IDs
    ids = list(data.keys())
    
    # Write IDs to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id'])  # header
        writer.writerows([[id_] for id_ in ids])

if __name__ == '__main__':
    extract_ids()
