import json
import os
from pathlib import Path
from typing import List, Dict, Any


def extract_housing_data(input_path: str, output_path: str) -> List[Dict[str, Any]]:
    """Extract housing data from demographics json and format it for agent json.
    
    Args:
        input_path: Path to input demographics json file
        output_path: Path to output agent json file
        
    Returns:
        List of formatted agent data dictionaries
    """
    # Read input demographics json
    with open(input_path, 'r', encoding='utf-8') as f:
        demo_data = json.load(f)
    
    results = []
    
    # Process each participant's data
    for participant_id, data in demo_data.items():
        # Create agent data structure
        participant_data = {
            "id": participant_id,
            "agent": {
                "age": data.get("age"),
                "Geo Mobility": data.get("moved_last_year"),
                "householder type": data.get("housing_status"),
                "Gross rent": data.get("rent_income_ratio"),
                "means of transportation": data.get("transportation"),
                "income": data.get("household_income"),
                "occupation": data.get("occupation"),
                "family type": data.get("marital_status"),
                "children": data.get("children_age"),
                "housing experience": data.get("housing_experience"),
                # Additional fields from demographics
                "race_ethnicity": data.get("race_ethnicity"),
                "financial_situation": data.get("financial_situation"),
                "neighborhood_safety": data.get("neighborhood_safety"),
                "health_insurance": data.get("health_insurance"),
                "education": data.get("education"),
                "citizenship": data.get("citizenship"),
                "zipcode": data.get("zipcode")
            }
        }
        
        # Clean up None values to empty strings
        for key in participant_data["agent"]:
            if participant_data["agent"][key] is None:
                participant_data["agent"][key] = ""
        
        results.append(participant_data)
    
    # Write to JSON file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, indent=2)
    
    print(f"Extraction complete. Data saved to {output_path}")
    print(f"Total records processed: {len(results)}")
    
    # Print the first record as an example
    if results:
        print("\nSample record:")
        print(json.dumps(results[0], indent=2))
    
    return results


def main():
    # Define paths
    base_dir = Path(__file__).parent
    
    # Input file path
    input_path = base_dir / "processed" / "extropolate_30ppl_demographics.json"
    
    # Output file path
    output_path = base_dir / "processed" / "extropolate_30ppl_agent.json"
    
    # Extract data
    extract_housing_data(input_path, output_path)

    # for no attention check :python processor.py --csv your_survey.csv --no-attention-check


if __name__ == "__main__":
    main() 