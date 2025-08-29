#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import copy
import shutil

# Define the paths
RAW_FILE_PATH = './raw/raw_upzoning_proposal.json'
OUTPUT_DIR = './processed/mock_proposals'

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Load the raw upzoning proposal
with open(RAW_FILE_PATH, 'r') as f:
    raw_proposal = json.load(f)

# Define the patterns and their corresponding height limits
patterns = {
    '1': {'range': (65, 300), 'variants': {'1': 80, '2': 140, '3': 300}},
    '2': {'range': (85, 300), 'variants': {'1': 80, '2': 140, '3': 300}},
    '3': {'range': (240, 300), 'variants': {'1': 80, '2': 140, '3': 300}}
}

# Create a mapping of cells to patterns
pattern_cells = {
    '1': [],  # Cells with height limits between 65 and 300
    '2': [],  # Cells with height limits between 85 and 300
    '3': []   # Cells with height limits between 240 and 300
}

# Classify cells into patterns based on height limits
for cell_id, cell_data in raw_proposal['cells'].items():
    height_limit = cell_data.get('heightLimit', 0)
    
    # Pattern 1: 65-300
    if 65 <= height_limit <= 300:
        pattern_cells['1'].append(cell_id)
    
    # Pattern 2: 85-300
    if 85 <= height_limit <= 300:
        pattern_cells['2'].append(cell_id)
    
    # Pattern 3: 240-300
    if 240 <= height_limit <= 300:
        pattern_cells['3'].append(cell_id)

# Generate the 9 mock proposals
for pattern_id, cells in pattern_cells.items():
    for variant_id, new_height in patterns[pattern_id]['variants'].items():
        # Create a copy of the raw proposal
        mock_proposal = copy.deepcopy(raw_proposal)
        
        # Create a new cells dictionary that only includes cells in the current pattern
        new_cells = {}
        
        # Only keep cells that are in the current pattern and update their height limits
        for cell_id in cells:
            if cell_id in mock_proposal['cells']:
                new_cells[cell_id] = mock_proposal['cells'][cell_id]
                new_cells[cell_id]['heightLimit'] = new_height
        
        # Replace the original cells dictionary with the filtered one
        mock_proposal['cells'] = new_cells
        
        # Save the mock proposal to a JSON file with the format "pattern.variant.json"
        filename = f"{pattern_id}.{variant_id}.json"
        output_file = os.path.join(OUTPUT_DIR, filename)
        with open(output_file, 'w') as f:
            json.dump(mock_proposal, f, indent=4)
        
        # Print status
        print(f"Generated mock proposal {filename} with {len(cells)} cells, all set to height {new_height}")

# Clean up any existing subdirectories
for item in os.listdir(OUTPUT_DIR):
    item_path = os.path.join(OUTPUT_DIR, item)
    if os.path.isdir(item_path):
        shutil.rmtree(item_path)
        print(f"Removed directory: {item_path}")

print("All mock proposals generated successfully!") 