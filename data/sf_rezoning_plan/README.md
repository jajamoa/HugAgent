# SF Rezoning Plan Data

This folder contains data related to San Francisco rezoning plans.

## Directory Structure

```
sf_rezoning_plan/
├── raw/              # Raw GeoJSON files and processing scripts
├── processed/        # Processed JSON files for the rezoning proposals
└── README.md         # This file
```

## Contents

### Raw Data

The raw data folder contains:
- GeoJSON files with SF zoning information
- Python scripts for downloading and converting the raw zoning data
- Documentation about the data sources

### Processed Data

The processed data folder contains:
- JSON files with processed rezoning proposals
- These files are used by the evaluation system

## Usage

The raw data can be processed into proposal formats using the scripts in the raw directory.

```bash
cd src/experiment/eval/data/sf_rezoning_plan/raw
python convert_raw_to_proposal.py
``` 