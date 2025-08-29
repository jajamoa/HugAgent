# Evaluation Data Samples

This directory contains sample files that demonstrate the data schema for evaluation metrics and model outputs.

## Files

- `metric_schema.json`: JSON Schema describing the data formats for ground truth, model predictions, and evaluation metrics
- `metric_schema_improved.json`: Improved JSON Schema with better structure and validation
- `sample_ground_truth.json`: Example ground truth data following the schema
- `sample_ground_truth_mapped.json`: Example ground truth with coded reason labels
- `sample_model_prediction.json`: Example model prediction data following the schema
- `sample_model_prediction_mapped.json`: Example model prediction with coded reason labels
- `sample_metrics_output.json`: Example metrics evaluation output following the schema
- `sample_proposal.json`: Example policy proposal data

## Data Format

The data format is designed to facilitate batch evaluation of housing policy opinion models:

### Ground Truth Format

Ground truth data includes both demographic information and reactions:

```json
{
  "<participant_id>": {
    "demographics": { ... },
    "reactions": {
      "opinions": {
        "<scenario_id>": <rating_1_to_10>
      },
      "reasons": {
        "<scenario_id>": ["reason1", "reason2", ...]
      }
    }
  }
}
```

### Model Prediction Format

Model predictions include only the reactions:

```json
{
  "<participant_id>": {
    "opinions": {
      "<scenario_id>": <predicted_rating_1_to_10>
    },
    "reasons": {
      "<scenario_id>": ["predicted_reason1", "predicted_reason2", ...]
    }
  }
}
```

### Mapped Format

For efficiency and standardization, reason labels can be represented using short code mappings:

```json
{
  "<participant_id>": {
    "opinions": {
      "<scenario_id>": <predicted_rating_1_to_10>
    },
    "reasons": {
      "<scenario_id>": ["A", "B", "C", ...]
    }
  }
}
```

The mapping between codes and full reason labels is defined in `reason_mapping.json`:

```json
{
  "mapping": {
    "Housing supply and availability": "A",
    "Affordability for low- and middle-income residents": "B",
    ...
  },
  "reverse_mapping": {
    "A": "Housing supply and availability",
    "B": "Affordability for low- and middle-income residents",
    ...
  }
}
```

### Metrics Format

Evaluation metrics compare ground truth to model predictions:

```json
{
  "opinion_metrics": {
    "rmse": <value>,
    "mae": <value>,
    "r2": <value>
  },
  "reason_metrics": {
    "jaccard_index": <value>,
    "micro_precision": <value>,
    "micro_recall": <value>,
    "micro_f1": <value>
  },
  "per_scenario": {
    "<scenario_id>": {
      "opinion_metrics": { ... },
      "reason_metrics": { ... }
    }
  }
}
```

## Usage

These sample files can be used as references for:

1. Implementing evaluation metrics
2. Formatting model outputs correctly
3. Aligning agent outputs with the expected format
4. Developing and testing evaluation pipelines 