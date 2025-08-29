import json
import random
import os
from pathlib import Path

def extract_housing_data(json_file_path, output_path):
    """
    Extract specific data from San Francisco Housing Survey demographics JSON file and format it into agent.json.
    
    Args:
        json_file_path (str): Path to the demographics JSON file
        output_path (str): Path to save the output agent.json file
        
    Returns:
        list: List of dictionaries containing formatted data
    """
    results = []
    
    # Valid San Francisco ZIP codes
    valid_sf_zip_codes = [
        "94124", "94127", "94131", "94133", "94132", "94133", "94134", "94102", 
        "94158", "94103", "94104", "94105", "94107", "94108", "94109", "94110", 
        "94111", "94112", "94114", "94115", "94116", "94117", "94118", "94121", 
        "94122", "94123", "94129", "94130"
    ]
    
    # Default coordinates if ZIP code is not valid
    default_coordinates = {
        "lat": 37.778250, 
        "lng": -122.419944
    }  # 37°46'41.7"N 122°25'11.8"W
    
    # ZIP code to coordinates mapping (approximate center points)
    zip_coordinates = {
        "94124": {"lat": 37.7299, "lng": -122.3854},  # Bayview
        "94127": {"lat": 37.7362, "lng": -122.4577},  # St. Francis Wood
        "94131": {"lat": 37.7436, "lng": -122.4375},  # Twin Peaks
        "94132": {"lat": 37.7219, "lng": -122.4783},  # Lake Merced
        "94133": {"lat": 37.8002, "lng": -122.4091},  # North Beach
        "94134": {"lat": 37.7149, "lng": -122.4130},  # Visitacion Valley
        "94102": {"lat": 37.7786, "lng": -122.4156},  # Tenderloin
        "94158": {"lat": 37.7704, "lng": -122.3880},  # Mission Bay
        "94103": {"lat": 37.7752, "lng": -122.4144},  # SoMa
        "94104": {"lat": 37.7915, "lng": -122.4022},  # Financial District
        "94105": {"lat": 37.7866, "lng": -122.3890},  # Rincon Hill
        "94107": {"lat": 37.7621, "lng": -122.3971},  # Potrero Hill
        "94108": {"lat": 37.7929, "lng": -122.4079},  # Chinatown
        "94109": {"lat": 37.7916, "lng": -122.4223},  # Nob Hill
        "94110": {"lat": 37.7485, "lng": -122.4158},  # Mission District
        "94111": {"lat": 37.7976, "lng": -122.4004},  # Embarcadero
        "94112": {"lat": 37.7200, "lng": -122.4369},  # Ingleside
        "94114": {"lat": 37.7599, "lng": -122.4346},  # Castro
        "94115": {"lat": 37.7857, "lng": -122.4358},  # Pacific Heights
        "94116": {"lat": 37.7435, "lng": -122.4892},  # Sunset
        "94117": {"lat": 37.7692, "lng": -122.4449},  # Haight-Ashbury
        "94118": {"lat": 37.7811, "lng": -122.4617},  # Richmond
        "94121": {"lat": 37.7786, "lng": -122.4892},  # Outer Richmond
        "94122": {"lat": 37.7599, "lng": -122.4828},  # Inner Sunset
        "94123": {"lat": 37.7991, "lng": -122.4340},  # Marina
        "94129": {"lat": 37.7983, "lng": -122.4701},  # Presidio
        "94130": {"lat": 37.8232, "lng": -122.3693}   # Treasure Island
    }
    
    # Add slight variation to coordinates to spread out points
    def add_coordinate_variation(base_coords):
        # Add a small random variation (approximately within 500 meters)
        lat_variation = random.uniform(-0.003, 0.003)  
        lng_variation = random.uniform(-0.003, 0.003)
        return {
            "lat": base_coords["lat"] + lat_variation,
            "lng": base_coords["lng"] + lng_variation
        }
    
    # Load demographics data
    with open(json_file_path, 'r', encoding='utf-8') as file:
        demographics = json.load(file)
    
    for participant_id, data in demographics.items():
        # Extract ZIP code and determine coordinates
        zipcode = data.get("What is your ZIP code?", "")  # Remove any trailing colons
        
        # Assign coordinates based on ZIP code
        if zipcode in valid_sf_zip_codes:
            # Use coordinates for this ZIP code with slight variation
            coordinates = add_coordinate_variation(zip_coordinates[zipcode])
        else:
            # Use default coordinates with slight variation
            coordinates = add_coordinate_variation(default_coordinates)
        
        # Map demographic data - only map age, keep everything else as original
        participant_data = {
            "id": participant_id,
            "coordinates": coordinates,
            "agent": {
                "age": data.get("What is your age?", ""),
                "Geo Mobility": data.get("Have you moved in the past year?", ""),
                "householder type": data.get("What best describes your housing status?", ""),
                "Gross rent": data.get("If you rent, what is your approximate monthly rent as a percentage of your income?", ""),
                "means of transportation": data.get("What is your primary mode of transportation? (Please select all that apply)", ""),
                "income": data.get("What is your annual household income?", ""),
                "occupation": data.get("Which of the following best describes your occupation?", ""),
                "family type": data.get("What's your marital status?", ""),
                "children": data.get("What is the age range of your children under 18 years old? (You may select more than one option.)", ""),
                "housing experience": data.get("In the past five years, briefly describe your housing experience in San Francisco, including any moves, rental situations, and changes in your housing status. What were the reasons for these changes?", ""),
            }
        }
        
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
    base_dir = Path(__file__).parents[1]  # sf_prolific_survey directory
    
    # Input file path
    input_path = base_dir / "processed" / "SF_filtered_paused_5.15_demographics.json"
    
    # Output file path
    output_path = base_dir / "processed" / "agent_5.15.json"
    
    # Extract data
    extract_housing_data(input_path, output_path)

if __name__ == "__main__":
    main() 