# HugAgent: A Benchmark for Human-Grounded Agent Beliefs in Realistic Social Contexts

A benchmark for evaluating LLM theory of mind capabilities in realistic social contexts using real human conversation data. The benchmark includes belief attribution and belief update tasks across multiple difficulty levels, designed to assess model performance on complex, long-term social reasoning scenarios.

## Usage

### Core Scripts

#### Data Processing
```bash
cd Benchmark/
python process_data.py
# Processes pilot_36users/ folder and generates sample.jsonl

# With options (space-separated context lengths)
python process_data.py --context-lengths short medium --max-workers 5 --max-users 20

# Single context length
python process_data.py --context-lengths long --task-type belief_attribution --topic zoning

# All context lengths (default)
python process_data.py --task-type belief_update --topic healthcare
```

#### Model Evaluation  
```bash
# Single model
python evaluate_qwen.py --model qwen-plus --benchmark_path sample_150.jsonl

# Debug mode (sequential, interactive)
python evaluate_qwen.py --model qwen-plus --debug

# Batch evaluation (all models in parallel)
./run_all_evaluations.sh
```

**Evaluation Options:**
- `--benchmark_path`: JSONL file path (default: sample.jsonl)
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

### Visualization Tool

Launch interactive analysis interface:
```bash
# Open in browser
open visualization.html

# Or serve locally
python -m http.server 8000
```

**Usage:**
1. Drag evaluation JSON + sample JSONL files
2. Select models and difficulty 
3. Click grid cells for question details

## Benchmark Structure

**Context Length Levels:**
- **Long**: Full context (all 20+ conversation turns)
- **Medium**: Partial context (10 conversation turns)  
- **Short**: Minimal context (5 conversation turns)

Each level tests belief attribution with decreasing information availability.

## Files

```
Benchmark/
├── process_data.py          # Extract QA pairs from pilot_36users/
├── evaluate_qwen.py         # Run model evaluation with ToM prompts
├── run_all_evaluations.sh   # Batch evaluation script (parallel)
├── llm_utils.py            # Qwen API wrapper with debug support
├── sample_150.jsonl        # Generated benchmark questions
├── pilot_36users/          # Raw participant data
├── evaluation_results_*.json # Model performance by difficulty
└── visualization.html      # Interactive analysis interface
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