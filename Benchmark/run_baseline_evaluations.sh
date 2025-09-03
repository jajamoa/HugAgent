#!/bin/bash

# Script to run baseline model evaluations on filtered benchmark datasets
# Usage: ./run_baseline_evaluations.sh

# List of filtered datasets to evaluate
BELIEF_ATTRIBUTION_DATASETS=(
    "sample_belief_attribution_healthcare_filtered_different_results.jsonl"
    "sample_belief_attribution_surveillance_filtered_different_results.jsonl"
    "sample_belief_attribution_zoning_filtered_different_results.jsonl"
)

BELIEF_UPDATE_DATASETS=(
    "sample_belief_update_healthcare_filtered_context_improves.jsonl"
    "sample_belief_update_surveillance_filtered_context_improves.jsonl"
    "sample_belief_update_zoning_filtered_context_improves.jsonl"
)

# Topics for each task
TOPICS=("healthcare" "surveillance" "zoning")

# List of baseline models
MODELS=(
    "global-majority"
    "random-guess"
)

# Create logs directory if it doesn't exist
mkdir -p logs

# Function to run evaluation for a single dataset
run_evaluation() {
    local dataset=$1
    local model=$2
    local task_type=$3
    local topic=$4
    
    if [ ! -f "$dataset" ]; then
        echo "Warning: Dataset $dataset not found, skipping..."
        return 1
    fi
    
    # Extract dataset name for log file
    dataset_name=$(basename "$dataset" .jsonl)
    log_file="logs/eval_${model}_${dataset_name}.log"
    results_file="evaluation_results_${model}_${dataset_name}.json"
    
    # Run evaluation (suppress output)
    python evaluate_baselines.py \
        --benchmark_path "$dataset" \
        --model "$model" \
        --seed 42 \
        > "$log_file" 2>&1
    
    return 0
}

# Function to calculate average metrics for a task
calculate_task_average() {
    local model=$1
    local task_type=$2
    
    echo "=== $task_type with $model ==="
    
    # Create Python script to calculate averages
    python3 -c "
import json
import sys
from pathlib import Path

model = '$model'
task_type = '$task_type'
topics = ['healthcare', 'surveillance', 'zoning']

results = {}
total_correct = 0
total_questions = 0
all_mae_errors = []

for topic in topics:
    if task_type == 'belief_attribution':
        filename = f'evaluation_results_{model}_sample_belief_attribution_{topic}_filtered_different_results.json'
    else:
        filename = f'evaluation_results_{model}_sample_belief_update_{topic}_filtered_context_improves.json'
    
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            
        # Sum up results across all difficulty levels
        topic_correct = 0
        topic_total = 0
        topic_mae_errors = []
        
        for difficulty, metrics in data.items():
            topic_correct += metrics.get('correct', 0)
            topic_total += metrics.get('total', 0)
            if metrics.get('absolute_errors'):
                topic_mae_errors.extend(metrics['absolute_errors'])
        
        topic_accuracy = topic_correct / topic_total if topic_total > 0 else 0
        topic_mae = sum(topic_mae_errors) / len(topic_mae_errors) if topic_mae_errors else None
        
        results[topic] = {
            'accuracy': topic_accuracy,
            'correct': topic_correct,
            'total': topic_total,
            'mae': topic_mae
        }
        
        total_correct += topic_correct
        total_questions += topic_total
        if topic_mae_errors:
            all_mae_errors.extend(topic_mae_errors)
            
        print(f'{topic.capitalize()}: {topic_accuracy:.1%} ({topic_correct}/{topic_total})', end='')
        if topic_mae is not None:
            print(f', MAE: {topic_mae:.2f}')
        else:
            print()
            
    except FileNotFoundError:
        print(f'Error: {filename} not found')
    except Exception as e:
        print(f'Error processing {topic}: {e}')

# Calculate overall average
overall_accuracy = total_correct / total_questions if total_questions > 0 else 0
overall_mae = sum(all_mae_errors) / len(all_mae_errors) if all_mae_errors else None

print(f'Average: {overall_accuracy:.1%} ({total_correct}/{total_questions})', end='')
if overall_mae is not None:
    print(f', MAE: {overall_mae:.2f}')
else:
    print()

# Save average results
avg_results = {
    'task_type': task_type,
    'model': model,
    'topics': results,
    'average': {
        'accuracy': overall_accuracy,
        'correct': total_correct,
        'total': total_questions,
        'mae': overall_mae
    }
}

avg_filename = f'evaluation_results_{model}_{task_type}_average.json'
with open(avg_filename, 'w') as f:
    json.dump(avg_results, f, indent=2)
"
    echo ""
}

# Run evaluations for each model
for model in "${MODELS[@]}"; do
    # Evaluate Belief Attribution datasets
    for i in "${!BELIEF_ATTRIBUTION_DATASETS[@]}"; do
        dataset="${BELIEF_ATTRIBUTION_DATASETS[$i]}"
        topic="${TOPICS[$i]}"
        run_evaluation "$dataset" "$model" "belief_attribution" "$topic"
    done
    
    # Calculate average for belief attribution
    calculate_task_average "$model" "belief_attribution"
    
    # Evaluate Belief Update datasets  
    for i in "${!BELIEF_UPDATE_DATASETS[@]}"; do
        dataset="${BELIEF_UPDATE_DATASETS[$i]}"
        topic="${TOPICS[$i]}"
        run_evaluation "$dataset" "$model" "belief_update" "$topic"
    done
    
    # Calculate average for belief update
    calculate_task_average "$model" "belief_update"
done
