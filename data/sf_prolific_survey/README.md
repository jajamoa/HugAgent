# SF Prolific Survey Data Processor

This folder contains a processor for CSV survey data from Prolific, which extracts demographic information and scenario-based reaction data (opinions + reasons) from participants.

## Directory Structure

```
sf_prolific_survey/
├── raw/              # Raw CSV files from Prolific surveys
├── processed/        # Processed JSON outputs (demographics and reactions)
├── processor.py      # Script to process the surveys
├── test_processor.py # Test script to validate processor output
└── README.md         # This file
```

## Usage

1. Place your CSV survey files in the `raw/` directory.
2. Run the processor script:

```bash
cd src/experiment/eval/data/sf_prolific_survey
python processor.py
```

3. The processed data will be saved in the `processed/` directory with the following naming convention:
   - `{original_filename}_demographics.json`: Contains demographic information with participant IDs
   - `{original_filename}_reactions.json`: Contains opinions and reasons with participant IDs

## Data Format

The processor specifically handles the SF Prolific Survey format:

- **Participant IDs**: Extracted from the "Prolific ID" column
- **Demographics**: Extracts standardized demographic fields including:
  - Age, housing status, income, occupation, transportation, household situation, etc.
  - Housing experience narrative
- **Opinions**: Extracts numeric ratings from scenario columns (e.g., "Scenario 1.1")
  - Stored as numeric values (1-10 scale)
- **Reasons**: Extracts selections from reason columns (e.g., "Scenario 1.1: Select the reasons")
  - Parsed into lists for multi-selection responses

The output is organized by participant ID for easy lookup and metric evaluation. 