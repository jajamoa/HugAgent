#!/bin/bash

# HugToM-QA Benchmark Evaluation Script
# Runs evaluation for multiple Qwen models in parallel

set -e  # Exit on any error

# Configuration
BENCHMARK_PATH="sample_150.jsonl"
TEMPERATURE=0.1
MAX_WORKERS=6
LOG_DIR="logs"

# Model list
MODELS=(
    "qwen-max"
    "qwen-plus" 
    "qwen2.5-32b-instruct"
    "qwen2.5-7b-instruct"
    "qwen2.5-0.5b-instruct"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create log directory
mkdir -p "$LOG_DIR"

# Function to run evaluation for a single model
run_evaluation() {
    local model=$1
    local log_file="$LOG_DIR/eval_${model}.log"
    local start_time=$(date +%s)
    
    echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} Starting evaluation for ${YELLOW}$model${NC}"
    
    # Run evaluation and capture both stdout and stderr
    if python evaluate_qwen.py \
        --benchmark_path "$BENCHMARK_PATH" \
        --model "$model" \
        --temperature "$TEMPERATURE" \
        --max-workers "$MAX_WORKERS" \
        > "$log_file" 2>&1; then
        
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} ✓ Completed ${YELLOW}$model${NC} in ${duration}s"
        
        # Extract accuracy from results file
        local results_file="evaluation_results_${model}_by_difficulty.json"
        if [[ -f "$results_file" ]]; then
            echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} Results saved to $results_file"
            
            # Extract and display accuracy summary
            if command -v jq >/dev/null 2>&1; then
                echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} Summary for $model:"
                jq -r '.simple.accuracy as $s | .medium.accuracy as $m | .hard.accuracy as $h | "  Simple: \($s*100|floor)%  Medium: \($m*100|floor)%  Hard: \($h*100|floor)%"' "$results_file"
            fi
        fi
    else
        echo -e "${RED}[$(date '+%H:%M:%S')]${NC} ✗ Failed ${YELLOW}$model${NC} (check $log_file)"
        return 1
    fi
}

# Main execution
main() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  HugToM-QA Benchmark Evaluation${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "${BLUE}Benchmark:${NC} $BENCHMARK_PATH"
    echo -e "${BLUE}Models:${NC} ${#MODELS[@]} models"
    echo -e "${BLUE}Temperature:${NC} $TEMPERATURE"
    echo -e "${BLUE}Max Workers:${NC} $MAX_WORKERS"
    echo -e "${BLUE}Log Directory:${NC} $LOG_DIR"
    echo ""
    
    # Check if benchmark file exists
    if [[ ! -f "$BENCHMARK_PATH" ]]; then
        echo -e "${RED}Error:${NC} Benchmark file '$BENCHMARK_PATH' not found!"
        exit 1
    fi
    
    # Check if Python script exists
    if [[ ! -f "evaluate_qwen.py" ]]; then
        echo -e "${RED}Error:${NC} evaluate_qwen.py not found!"
        exit 1
    fi
    
    local overall_start=$(date +%s)
    local pids=()
    local failed_models=()
    
    # Start all evaluations in parallel
    for model in "${MODELS[@]}"; do
        run_evaluation "$model" &
        pids+=($!)
    done
    
    echo -e "${YELLOW}[$(date '+%H:%M:%S')]${NC} All evaluations started, waiting for completion..."
    echo ""
    
    # Wait for all processes and check results
    for i in "${!pids[@]}"; do
        local pid=${pids[$i]}
        local model=${MODELS[$i]}
        
        if wait $pid; then
            echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} Process for $model completed successfully"
        else
            echo -e "${RED}[$(date '+%H:%M:%S')]${NC} Process for $model failed"
            failed_models+=("$model")
        fi
    done
    
    local overall_end=$(date +%s)
    local total_duration=$((overall_end - overall_start))
    
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Evaluation Summary${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "${BLUE}Total Duration:${NC} ${total_duration}s"
    echo -e "${BLUE}Successful:${NC} $((${#MODELS[@]} - ${#failed_models[@]}))/${#MODELS[@]} models"
    
    if [[ ${#failed_models[@]} -gt 0 ]]; then
        echo -e "${RED}Failed Models:${NC} ${failed_models[*]}"
        echo -e "${YELLOW}Check log files in $LOG_DIR/ for details${NC}"
    fi
    
    echo ""
    echo -e "${BLUE}Result Files:${NC}"
    for model in "${MODELS[@]}"; do
        local results_file="evaluation_results_${model}_by_difficulty.json"
        if [[ -f "$results_file" ]]; then
            local size=$(ls -lh "$results_file" | awk '{print $5}')
            echo -e "  ✓ $results_file ($size)"
        else
            echo -e "  ✗ $results_file ${RED}(missing)${NC}"
        fi
    done
    
    echo ""
    echo -e "${GREEN}Use visualization.html to analyze results${NC}"
    
    # Exit with error if any evaluations failed
    if [[ ${#failed_models[@]} -gt 0 ]]; then
        exit 1
    fi
}

# Handle interruption
trap 'echo -e "\n${RED}Interrupted! Killing background processes...${NC}"; jobs -p | xargs -r kill; exit 130' INT TERM

# Run main function
main "$@"
