#!/usr/bin/env python3
"""
Extract demographic data for each user and save to their respective folders.
"""

import json
import os
from pathlib import Path

def extract_user_demographics():
    # Load demographic data
    with open("data/surveys_100/survey_96ppl_demographics.json", "r") as f:
        demographics_data = json.load(f)
    
    # Get user IDs from pilot_5users directory
    pilot_dir = Path("pilot_5users")
    user_dirs = [d for d in pilot_dir.iterdir() if d.is_dir() and len(d.name) >= 20]
    
    print(f"Found {len(user_dirs)} user directories")
    
    # Extract and save demographic data for each user
    for user_dir in user_dirs:
        user_id = user_dir.name
        
        # Check if user exists in demographic data
        if user_id in demographics_data:
            # Create demographic folder if it doesn't exist
            demo_folder = user_dir / "demographic"
            demo_folder.mkdir(exist_ok=True)
            
            # Extract user's demographic data
            user_demo = {user_id: demographics_data[user_id]}
            
            # Save to demographic.json
            demo_file = demo_folder / "demographic.json"
            with open(demo_file, "w") as f:
                json.dump(user_demo, f, indent=2)
            
            print(f"Saved demographic data for: {user_id}")
        else:
            print(f"User {user_id} not found in demographic data")

if __name__ == "__main__":
    extract_user_demographics()
