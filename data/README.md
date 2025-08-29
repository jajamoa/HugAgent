# Experiment Data

This directory contains data files for simulation and evaluation of agent opinions on housing policies.

## Directory Structure

```
data/
├── sf_prolific_survey/      # Survey data from participants
├── sf_rezoning_plan/        # Zoning proposal data
├── samples/                 # Sample data for testing
└── ... (other data sets)
```

## Data Formats

### Proposal Format (JSON)

Zoning proposals are stored in JSON format with the following structure:

```json
{
  "grid_config": {
    "cellSize": 100,
    "bounds": { ... }
  },
  "height_limits": {
    "default": 40,
    "options": [40, 65, 80, ...]
  },
  "cells": {
    "cell_id": {
      "height_limit": 65,
      "category": "residential",
      "bbox": { ... }
    },
    ...
  }
}
```

### Ground Truth Format (JSON)

Survey responses (ground truth) are stored in JSON with this structure:

```json
{
  "user_id": {
    "opinions": {
      "cell_id": 8,  // Scale: 1-10
      ...
    },
    "reasons": {
      "cell_id": ["transit", "housing_supply"],
      ...
    }
  },
  ...
}
```

## Using the Data

In experiment protocols, reference these files as:

```yaml
input:
  proposals:
    - "sf_rezoning_plan/processed/proposal_abc.json" 

evaluation:
  ground_truth: "sf_prolific_survey/processed/survey_responses.json"
```

## Adding New Data

To add new data:
1. Create appropriately named subdirectories
2. Document data sources and processing steps
3. Ensure the output format matches the expected structure 