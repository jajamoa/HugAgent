#!/usr/bin/env python3
"""
Check if all files are properly equipped and aligned for each individual user folder.
"""

import json
import csv
import os
from pathlib import Path
from collections import defaultdict

def check_user_files():
    pilot_dir = Path("pilot_5users")
    user_dirs = [d for d in pilot_dir.iterdir() if d.is_dir() and len(d.name) >= 20]
    
    print(f"Checking {len(user_dirs)} user directories...\n")
    
    missing_files = defaultdict(list)
    stats = {
        "total_users": len(user_dirs),
        "complete_users": 0,
        "missing_demographic": 0,
        "missing_survey": 0,
        "missing_transcript": 0
    }
    
    for user_dir in user_dirs:
        user_id = user_dir.name
        issues = []
        
        # Check demographic folder
        demo_folder = user_dir / "demographic"
        demo_file = demo_folder / "demographic.json"
        
        if not demo_folder.exists():
            issues.append("demographic folder missing")
            stats["missing_demographic"] += 1
        elif not demo_file.exists():
            issues.append("demographic.json missing")
            stats["missing_demographic"] += 1
        else:
            # Check if user ID in demographic.json matches folder name
            try:
                with open(demo_file, "r") as f:
                    demo_data = json.load(f)
                    if user_id not in demo_data:
                        issues.append("demographic.json user ID mismatch")
                        stats["missing_demographic"] += 1
            except:
                issues.append("demographic.json corrupted")
                stats["missing_demographic"] += 1
        
        # Check survey folder
        survey_folder = user_dir / "survey"
        survey_files = ["camera_reaction.json", "healthcare_reaction.json", "zoning_reaction.json"]
        
        if not survey_folder.exists():
            issues.append("survey folder missing")
            stats["missing_survey"] += 1
        else:
            for survey_file in survey_files:
                file_path = survey_folder / survey_file
                if not file_path.exists():
                    issues.append(f"{survey_file} missing")
                    stats["missing_survey"] += 1
                else:
                    # Check if user ID in survey file matches folder name
                    try:
                        with open(file_path, "r") as f:
                            survey_data = json.load(f)
                            if user_id not in survey_data:
                                issues.append(f"{survey_file} user ID mismatch")
                                stats["missing_survey"] += 1
                    except:
                        issues.append(f"{survey_file} corrupted")
                        stats["missing_survey"] += 1
        
        # Check transcript folder
        transcript_folder = user_dir / "transcript"
        raw_folder = transcript_folder / "raw"
        processed_folder = transcript_folder / "processed"
        
        if not transcript_folder.exists():
            issues.append("transcript folder missing")
            stats["missing_transcript"] += 1
        elif not raw_folder.exists():
            issues.append("transcript/raw folder missing")
            stats["missing_transcript"] += 1
        else:
            # Check raw CSV files
            csv_files = list(raw_folder.glob("*.csv"))
            if len(csv_files) == 0:
                issues.append("no CSV files in transcript/raw")
                stats["missing_transcript"] += 1
            else:
                # Check if prolific ID in CSV matches folder name
                for csv_file in csv_files:
                    try:
                        with open(csv_file, "r") as f:
                            reader = csv.DictReader(f)
                            if reader.fieldnames and "prolific_id" in reader.fieldnames:
                                for row in reader:
                                    if row.get("prolific_id") and row["prolific_id"] != user_id:
                                        issues.append(f"CSV prolific_id mismatch in {csv_file.name}")
                                        stats["missing_transcript"] += 1
                                        break
                    except:
                        issues.append(f"CSV file corrupted: {csv_file.name}")
                        stats["missing_transcript"] += 1
        
        # Check processed folder (should be empty but exist)
        if transcript_folder.exists() and not processed_folder.exists():
            issues.append("transcript/processed folder missing")
            stats["missing_transcript"] += 1
        
        # Record issues
        if issues:
            missing_files[user_id] = issues
        else:
            stats["complete_users"] += 1
    
    # Print results
    print("=== FILE CHECK RESULTS ===\n")
    
    if missing_files:
        print("USERS WITH MISSING/INCONSISTENT FILES:")
        for user_id, issues in missing_files.items():
            print(f"\n{user_id}:")
            for issue in issues:
                print(f"  - {issue}")
    else:
        print("✅ All users have complete and consistent files!")
    
    print(f"\n=== STATISTICS ===")
    print(f"Total users: {stats['total_users']}")
    print(f"Complete users: {stats['complete_users']}")
    print(f"Users missing demographic: {stats['missing_demographic']}")
    print(f"Users missing survey: {stats['missing_survey']}")
    print(f"Users missing transcript: {stats['missing_transcript']}")
    print(f"Completion rate: {(stats['complete_users']/stats['total_users']*100):.1f}%")

if __name__ == "__main__":
    check_user_files()
