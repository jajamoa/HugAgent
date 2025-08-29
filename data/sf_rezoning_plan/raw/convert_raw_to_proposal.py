import json
import geopandas as gpd
from datetime import datetime
import numpy as np
import os
from shapely.geometry import box
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

def get_cell_bbox(row, col, bounds, grid_width, grid_height):
    """Get the bounding box for a grid cell.
    
    Note: row 0 starts from the top (north)
    """
    # Calculate cell boundaries using the same logic as frontend
    west = bounds["west"] + (col / grid_width) * (bounds["east"] - bounds["west"])
    east = bounds["west"] + ((col + 1) / grid_width) * (bounds["east"] - bounds["west"])
    
    # Match frontend's south-north calculation
    south = bounds["south"] + ((grid_height - row - 1) / grid_height) * (bounds["north"] - bounds["south"])
    north = bounds["south"] + ((grid_height - row) / grid_height) * (bounds["north"] - bounds["south"])
    
    return {
        "north": north,
        "south": south,
        "east": east,
        "west": west
    }

def process_cell(args, zoning_data):
    """Process a single grid cell. Find the largest intersecting polygon and use its height."""
    row, col, bounds, grid_width, grid_height = args
    
    # Get cell bbox
    cell_bbox = get_cell_bbox(row, col, bounds, grid_width, grid_height)
    cell_polygon = box(cell_bbox["west"], cell_bbox["south"], 
                      cell_bbox["east"], cell_bbox["north"])
    
    max_area = 0
    cell_info = None
    
    # Check all intersecting zones and find the one with largest intersection
    for _, zone in zoning_data.iterrows():
        if zone.geometry.intersects(cell_polygon):
            intersection = zone.geometry.intersection(cell_polygon)
            area = intersection.area
            if area > max_area:
                max_area = area
                cell_info = {
                    "heightLimit": zone["NEW_HEIGHT_NUM"],  # Using 2024 height data
                    "category": "mixed_use",  # Default category
                    "bbox": cell_bbox,
                    "area": area
                }
    
    if cell_info:
        return (row, col), cell_info
    return None

def main():
    print("Starting conversion process...")
    
    # Set file paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, "sf_zoning_2024.geojson")
    output_file = os.path.join(current_dir, "sf_proposal_2024.json")

    # Read 2024 zoning data
    print("Reading zoning data...")
    zoning_data = gpd.read_file(input_file)
    print(f"Found {len(zoning_data)} zones")
    
    # Get unique height values from the data
    height_options = sorted(zoning_data["NEW_HEIGHT_NUM"].unique().tolist())
    print(f"Found height options: {height_options}")
    
    # Define grid
    cell_size_meters = 100  # Cell size in meters
    bounds = {
        "north": 37.8120,
        "south": 37.7080,
        "east": -122.3549,
        "west": -122.5157
    }
    
    # Calculate grid dimensions using the same logic as frontend
    avg_lat = (bounds["north"] + bounds["south"]) / 2
    # Convert degrees to meters
    width = (bounds["east"] - bounds["west"]) * 111319.9 * np.cos(np.deg2rad(avg_lat))
    height = (bounds["north"] - bounds["south"]) * 111319.9
    
    # Calculate grid dimensions
    grid_width = int(np.floor(width / cell_size_meters))
    grid_height = int(np.floor(height / cell_size_meters))
    
    print(f"Grid dimensions: {grid_width}x{grid_height} cells")
    
    # Create proposal structure
    proposal = {
        "gridConfig": {
            "cellSize": cell_size_meters,
            "bounds": bounds
        },
        "heightLimits": {
            "default": 0,
            "options": height_options  # Use actual height options from data
        },
        "cells": {}
    }
    
    # Create list of all grid coordinates
    grid_coords = [(row, col, bounds, grid_width, grid_height) 
                  for row in range(grid_height) 
                  for col in range(grid_width)]
    
    # Set up multiprocessing
    num_processes = mp.cpu_count()
    print(f"Using {num_processes} processes")
    
    # Create partial function with fixed arguments
    process_cell_partial = partial(process_cell, zoning_data=zoning_data)
    
    # Process grid cells in parallel
    print("Processing grid cells in parallel...")
    with mp.Pool(num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_cell_partial, grid_coords),
            total=len(grid_coords),
            desc="Processing cells"
        ))
    
    # Create cells from results
    print("Creating final proposal...")
    for result in tqdm(results, desc="Creating cells"):
        if result is not None:
            (row, col), cell_info = result
            cell_id = f"{row}_{col}"
            proposal["cells"][cell_id] = {
                "heightLimit": cell_info["heightLimit"],
                "category": cell_info["category"],
                "lastUpdated": datetime.now().strftime("%Y-%m-%d"),
                "bbox": cell_info["bbox"]
            }
    
    # Save results
    print("Saving results...")
    with open(output_file, "w") as f:
        json.dump(proposal, f, indent=4)
    
    print(f"Conversion complete! Created {len(proposal['cells'])} cells")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    main() 