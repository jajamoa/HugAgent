import csv
import json
import random

def extract_housing_data(csv_file_path):
    """
    Extract demographic data from San Francisco Housing Survey CSV file and format it into JSON.
    
    Args:
        csv_file_path (str): Path to the CSV file
        
    Returns:
        list: List of dictionaries containing formatted data
    """
    results = []

    # Valid San Francisco ZIP codes
    valid_sf_zip_codes = [
        "94124", "94127", "94131", "94133", "94132", "94134", "94102", 
        "94158", "94103", "94104", "94105", "94107", "94108", "94109", "94110", 
        "94111", "94112", "94114", "94115", "94116", "94117", "94118", "94121", 
        "94122", "94123", "94129", "94130"
    ]

    # Default coordinates
    default_coordinates = {"lat": 37.778250, "lng": -122.419944}

    # ZIP code coordinate mapping
    zip_coordinates = {
        "94124": {"lat": 37.7299, "lng": -122.3854},
        "94127": {"lat": 37.7362, "lng": -122.4577},
        "94131": {"lat": 37.7436, "lng": -122.4375},
        "94132": {"lat": 37.7219, "lng": -122.4783},
        "94133": {"lat": 37.8002, "lng": -122.4091},
        "94134": {"lat": 37.7149, "lng": -122.4130},
        "94102": {"lat": 37.7786, "lng": -122.4156},
        "94158": {"lat": 37.7704, "lng": -122.3880},
        "94103": {"lat": 37.7752, "lng": -122.4144},
        "94104": {"lat": 37.7915, "lng": -122.4022},
        "94105": {"lat": 37.7866, "lng": -122.3890},
        "94107": {"lat": 37.7621, "lng": -122.3971},
        "94108": {"lat": 37.7929, "lng": -122.4079},
        "94109": {"lat": 37.7916, "lng": -122.4223},
        "94110": {"lat": 37.7485, "lng": -122.4158},
        "94111": {"lat": 37.7976, "lng": -122.4004},
        "94112": {"lat": 37.7200, "lng": -122.4369},
        "94114": {"lat": 37.7599, "lng": -122.4346},
        "94115": {"lat": 37.7857, "lng": -122.4358},
        "94116": {"lat": 37.7435, "lng": -122.4892},
        "94117": {"lat": 37.7692, "lng": -122.4449},
        "94118": {"lat": 37.7811, "lng": -122.4617},
        "94121": {"lat": 37.7786, "lng": -122.4892},
        "94122": {"lat": 37.7599, "lng": -122.4828},
        "94123": {"lat": 37.7991, "lng": -122.4340},
        "94129": {"lat": 37.7983, "lng": -122.4701},
        "94130": {"lat": 37.8232, "lng": -122.3693}
    }

    def add_coordinate_variation(base_coords):
        """Add small variation to coordinates (approx. ±500m)"""
        lat_variation = random.uniform(-0.003, 0.003)
        lng_variation = random.uniform(-0.003, 0.003)
        return {
            "lat": base_coords["lat"] + lat_variation,
            "lng": base_coords["lng"] + lng_variation
        }

    with open(csv_file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        for row in reader:
            try:
                zip_code_str = row["What is your ZIP code?"].strip()
                zip_code = str(int(zip_code_str))  # make sure it's normalized (no decimals, no colons)

                if zip_code in valid_sf_zip_codes:
                    coordinates = add_coordinate_variation(zip_coordinates[zip_code])
                else:
                    coordinates = add_coordinate_variation(default_coordinates)

                data = {
                    "id": row["Prolific ID"],
                    "coordinates": coordinates,
                    "agent": {
                        "age": int(row["What is your age?"]),
                        "Geo Mobility": row["Have you moved in the past year?"],
                        "householder type": row["What best describes your housing status?"],
                        "Gross rent": row["If you rent, what is your approximate monthly rent as a percentage of your income?"],
                        "means of transportation": row["What is your primary mode of transportation? (Please select all that apply)"],
                        "income": row["What is your annual household income?"],
                        "occupation": row["Which of the following best describes your occupation?"],
                        "marital status": row["What's your marital status?"],
                        "has children under 18": "Yes" in row["Do you have any children under the age of 18 living with you?"],
                        "children age range": row["What is the age range of your children under 18 years old? (You may select more than one option.)"],
                        "ZIP code": int(row["What is your ZIP code?"]),
                        "approximate address": row["What's your home address?\nIf you are not comfortable of providing the accurate location, you don’t have to give your exact address. Just something nearby is great — like a street corner, the name of a shop, or your building (no unit number needed). Anywhere within 50 meters of your home works!"]
                    }
                }

                results.append(data)

            except Exception as e:
                print(f"⚠️ Skipping row due to error: {e}")

    return results

def main():
    # Define the path to your CSV file
    csv_file_path = "./SF_survey_5.11.csv"  # TODO: change to the path of the csv file
    
    # Extract data
    extracted_data = extract_housing_data(csv_file_path)
    
    # Write to JSON file
    with open("responses_5.11.json", 'w', encoding='utf-8') as json_file:
        json.dump(extracted_data, json_file, indent=4)
    
    print(f"Extraction complete. Data saved to responses_5.11.json")
    print(f"Total records processed: {len(extracted_data)}")
    
    # Print the first record as an example
    if extracted_data:
        print("\nSample record:")
        print(json.dumps(extracted_data[0], indent=4))

if __name__ == "__main__":
    main()
