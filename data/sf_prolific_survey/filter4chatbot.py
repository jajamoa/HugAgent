import json

def filter_agents_by_ids(base_path: str, target_path: str, output_path: str = "filtered.json"):
    # Read the target ID list
    with open(target_path, 'r', encoding='utf-8') as f:
        # Handle both JSON array or plain text (line-separated IDs)
        try:
            data = json.load(f)
            target_ids = data if isinstance(data, list) else str(data).split()
        except json.JSONDecodeError:
            f.seek(0)
            target_ids = f.read().split()

    # Clean up any whitespace or empty lines
    target_ids = [id_.strip() for id_ in target_ids if id_.strip()]
    print(f"Total target IDs: {len(target_ids)}")

    # Load the base data (full agent records)
    with open(base_path, 'r', encoding='utf-8') as f:
        base_data = json.load(f)

    # Filter base data to only include records matching target IDs
    filtered = [item for item in base_data if item.get("id") in target_ids]
    print(f"Matched IDs found: {len(filtered)}")

    # Save the filtered results to the output file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(filtered)} records to file: {output_path}")

# Example usage
filter_agents_by_ids("processed/agent_5.15_with_geo.json", "target_id.json")
