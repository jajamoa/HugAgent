#!/usr/bin/env python3
"""
Extract individual user reactions from survey data and save to user folders.
"""

import json
import os
from pathlib import Path

def extract_user_reactions():
    # Load user IDs
    with open("pilot_5users/userid.txt", "r") as f:
        user_ids = [line.strip() for line in f if line.strip()]
    
    # Load survey data
    survey_data_dir = Path("data/surveys_100/processed")
    
    with open(survey_data_dir / "survey_96ppl_camera_reactions.json", "r") as f:
        camera_data = json.load(f)
    
    with open(survey_data_dir / "survey_96ppl_healthcare_reactions.json", "r") as f:
        healthcare_data = json.load(f)
    
    with open(survey_data_dir / "survey_96ppl_zoning_reactions.json", "r") as f:
        zoning_data = json.load(f)
    
    # Extract reactions for each user
    for user_id in user_ids:
        user_dir = Path(f"pilot_5users/{user_id}/survey")
        user_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract camera reactions
        if user_id in camera_data:
            camera_reaction = {user_id: camera_data[user_id]}
            with open(user_dir / "camera_reaction.json", "w") as f:
                json.dump(camera_reaction, f, indent=2)
            print(f"Saved camera reaction for {user_id}")
        
        # Extract healthcare reactions
        if user_id in healthcare_data:
            healthcare_reaction = {user_id: healthcare_data[user_id]}
            with open(user_dir / "healthcare_reaction.json", "w") as f:
                json.dump(healthcare_reaction, f, indent=2)
            print(f"Saved healthcare reaction for {user_id}")
        
        # Extract zoning reactions
        if user_id in zoning_data:
            zoning_reaction = {user_id: zoning_data[user_id]}
            with open(user_dir / "zoning_reaction.json", "w") as f:
                json.dump(zoning_reaction, f, indent=2)
            print(f"Saved zoning reaction for {user_id}")

if __name__ == "__main__":
    extract_user_reactions()
