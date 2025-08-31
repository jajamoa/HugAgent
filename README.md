# HugToM-QA: Towards Human-grounded Theory of Mind Question Answering

A benchmark for evaluating LLM theory of mind capabilities through human-grounded question answering tasks. Tests belief inference across three difficulty levels using real participant conversation data.

## Usage

### Core Scripts

#### Data Processing
```bash
cd Benchmark/
python process_data.py
# Processes pilot_36users/ folder and generates sample.jsonl
```

#### Model Evaluation  
```bash
python evaluate_qwen.py [OPTIONS]

# Examples:
python evaluate_qwen.py --model qwen-plus --benchmark_path sample_150.jsonl
python evaluate_qwen.py --model qwen2.5-7b-instruct --temperature 0.1
python evaluate_qwen.py --model qwen-max --no-demographics --no-context
```

**Options:**
- `--benchmark_path`: JSONL file path (default: sample.jsonl)
- `--model`: Model name (qwen-plus, qwen-max, qwen2.5-*, etc.)
- `--temperature`: Generation temperature (default: 0.1)  
- `--no-demographics`: Exclude participant demographics
- `--no-context`: Exclude conversation context
- `--max-workers`: Parallel workers (default: 3)

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

**Difficulty Levels:**
- **Simple**: Full context (all 20+ conversation turns)
- **Medium**: Partial context (10 conversation turns)  
- **Hard**: Minimal context (5 conversation turns)

Each level tests belief inference with decreasing information availability.

## Files

```
Benchmark/
├── process_data.py          # Extract QA pairs from pilot_36users/
├── evaluate_qwen.py         # Run model evaluation with ToM prompts
├── llm_utils.py            # Qwen API wrapper
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