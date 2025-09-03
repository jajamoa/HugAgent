# HugAgent: Human-Grounded Benchmarking of Agent Beliefs in Social Contexts

The first benchmark that evaluates models on belief attribution and belief updating using real participant surveys and interviews in socially salient domains (healthcare, surveillance, housing). Each instance combines demographic background with first-person self-reports, preserving individual variation instead of collapsing it into a single label.

## Usage

### Core Scripts

#### Data Processing
```bash
cd benchmark/
python process_data.py
# Processes ../raw_data/main_raw_data/ folder and generates sample.jsonl

# With options (space-separated context lengths)
python process_data.py --context-lengths short medium --max-workers 5 --max-users 20

# Single context length
python process_data.py --context-lengths long --task-type belief_attribution --topic zoning

# All context lengths (default)
python process_data.py --task-type belief_update --topic healthcare
```

#### Model Evaluation  
```bash
cd benchmark/
# Single model
python evaluate_qwen.py --model qwen-plus --benchmark_path sample_belief_attribution_healthcare.jsonl

# Debug mode (sequential, interactive)
python evaluate_qwen.py --model qwen-plus --debug

# Batch evaluation (all models in parallel)
./run_all_evaluations.sh
```

**Evaluation Options:**
- `--benchmark_path`: JSONL file path (default: sample_belief_attribution_healthcare.jsonl)
- `--model`: qwen-max, qwen-plus, qwen-turbo, qwen2.5-{72b,32b,14b,7b,3b,1.5b,0.5b}-instruct
- `--temperature`: Generation temperature 0.0-1.0 (default: 0.1)
- `--no-demographics`: Exclude participant demographics from prompt
- `--no-context`: Exclude context QAs from prompt  
- `--max-workers`: Parallel workers for API calls (default: 3)
- `--debug`: Sequential processing with full prompt/response display
- `--swap-experiment`: Use next participant's data for prediction (extreme test)

**Data Processing Options:**
- `--task-type`: Task type (choices: belief_attribution, belief_update; default: belief_attribution)
- `--topic`: Topic to process (choices: zoning, surveillance, healthcare; default: zoning)
- `--context-lengths`: Context lengths to generate (choices: short, medium, long; default: all; multiple values separated by spaces)
- `--max-workers`: Maximum parallel workers (default: 6)
- `--max-users`: Maximum user folders to process (default: 10)

### Human Annotation Tool

Launch interactive annotation interface:
```bash
# Open in browser
open human_annotation_tool.html

# Or serve locally
python -m http.server 8000
```

**Usage:**
1. Drag JSONL data files (belief attribution or belief update)
2. Select context length and start annotation
3. Export your annotations as JSON for analysis

## Benchmark Tasks

**Belief Attribution**: Inferring latent beliefs from conversational cues and participant responses. Models predict whether a person believes one factor has a positive, negative, or no significant effect on another.

**Belief Update**: Predicting how participants would change their opinions and reason weights under counterfactual interventions or scenarios.

## Files

```
├── raw_data/
│   ├── main_raw_data/           # Main participant data
│   ├── aux_raw_data/            # Auxiliary participant data  
│   └── survey_content/          # Survey questions and mappings
├── benchmark/
│   ├── process_data.py          # Extract QA pairs from ../raw_data/main_raw_data/
│   ├── evaluate_qwen.py         # Run model evaluation with ToM prompts
│   ├── run_all_evaluations.sh   # Batch evaluation script (parallel)
│   ├── llm_utils.py            # Qwen API wrapper with debug support
│   ├── sample_belief_*.jsonl    # Generated benchmark questions
│   └── evaluation_results_*.json # Model performance results
└── human_annotation_tool.html   # Interactive annotation interface
```

## Output Format

Evaluation generates `evaluation_results_{model}_by_difficulty.json`:
```json
{
  "simple": {"total": 50, "correct": 39, "accuracy": 0.78, "answers": [...]},
  "medium": {"total": 50, "correct": 40, "accuracy": 0.80, "answers": [...]},
  "hard":   {"total": 50, "correct": 34, "accuracy": 0.68, "answers": [...]}
}
```