#!/bin/bash

# HugToM-QA Benchmark Evaluation Script
# Runs evaluation for multiple Qwen models in parallel

set -e  # Exit on any error

# Configuration
# BENCHMARK_PATH="sample_belief_attribution_zoning.jsonl"
# BENCHMARK_PATH="sample_belief_attribution_zoning_filtered_different_results.jsonl"
# BENCHMARK_PATH="sample_belief_attribution_surveillance.jsonl"
# BENCHMARK_PATH="sample_belief_attribution_surveillance_filtered_different_results.jsonl"
# BENCHMARK_PATH="sample_belief_attribution_healthcare.jsonl"
# BENCHMARK_PATH="sample_belief_attribution_healthcare_filtered_different_results.jsonl"
# BENCHMARK_PATH="sample_belief_update_zoning.jsonl"
BENCHMARK_PATH="sample_belief_update_zoning_filtered_both_wrong_context_improves.jsonl" 

TEMPERATURE=0.1
MAX_WORKERS=6
LOG_DIR="logs"

# Model list
MODELS=(
    "qwen-max"
    "qwen-plus" 
    "qwen2.5-32b-instruct"
    "qwen2.5-7b-instruct"
    # "qwen2.5-0.5b-instruct"
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
    python evaluate_qwen.py \
        --benchmark_path "$BENCHMARK_PATH" \
        --model "$model" \
        --temperature "$TEMPERATURE" \
        --max-workers "$MAX_WORKERS" \
        > "$log_file" 2>&1
    
    local exit_code=$?
    local results_file="evaluation_results_${model}.json"
    
    # Check success based on both exit code and results file existence
    if [[ $exit_code -eq 0 ]] && [[ -f "$results_file" ]]; then
        
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} ✓ Completed ${YELLOW}$model${NC} in ${duration}s"
        
        # Display results file info
        if [[ -f "$results_file" ]]; then
            echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} Results saved to $results_file"
            
            # Extract and display accuracy/MAE summary
            if command -v jq >/dev/null 2>&1; then
                echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} Summary for $model:"
                # Check if this is belief_update (has MAE) or belief_attribution (has accuracy)
                local has_mae=$(jq -r 'if (.all.mae // null) != null or (.long.mae // null) != null or (.short.mae // null) != null then "true" else "false" end' "$results_file")
                
                if [[ "$has_mae" == "true" ]]; then
                    # Belief update format - show MAE
                    jq -r '(.all.mae // null) as $mae_all | (.all.accuracy // null) as $acc_all | (.long.mae // null) as $mae_l | (.long.accuracy // null) as $acc_l | (.short.mae // null) as $mae_s | (.short.accuracy // null) as $acc_s | if $mae_all != null and $acc_all != null then "  Overall: \($acc_all*100|floor)%, MAE: \($mae_all|.*1000|floor/1000)" elif $mae_l != null and $acc_l != null and $mae_s != null and $acc_s != null then "  Long: \($acc_l*100|floor)%, MAE: \($mae_l|.*1000|floor/1000)  Short: \($acc_s*100|floor)%, MAE: \($mae_s|.*1000|floor/1000)" elif $mae_l != null and $acc_l != null then "  Long: \($acc_l*100|floor)%, MAE: \($mae_l|.*1000|floor/1000)" elif $mae_s != null and $acc_s != null then "  Short: \($acc_s*100|floor)%, MAE: \($mae_s|.*1000|floor/1000)" else "  No performance data available" end' "$results_file"
                else
                    # Belief attribution format - show accuracy only
                    jq -r '(.all.accuracy // null) as $a | (.long.accuracy // null) as $l | (.short.accuracy // null) as $s | if $a != null then "  Overall: \($a*100|floor)%" elif $l != null and $s != null then "  Long: \($l*100|floor)%  Short: \($s*100|floor)%" elif $l != null then "  Long: \($l*100|floor)%" elif $s != null then "  Short: \($s*100|floor)%" else "  No accuracy data available" end' "$results_file"
                fi
            fi
        fi
    else
        echo -e "${RED}[$(date '+%H:%M:%S')]${NC} ✗ Failed ${YELLOW}$model${NC} (exit code: $exit_code, results file: $(test -f "$results_file" && echo "exists" || echo "missing"), check $log_file)"
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
    
    # Wait for all processes and check results with timeout
    local timeout_seconds=1800  # 30 minutes timeout
    local wait_start=$(date +%s)
    
    for i in "${!pids[@]}"; do
        local pid=${pids[$i]}
        local model=${MODELS[$i]}
        
        echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} Waiting for $model (PID: $pid)..."
        
        # Wait with timeout using background monitoring
        local wait_success=false
        while true; do
            if ! kill -0 "$pid" 2>/dev/null; then
                # Process has ended
                wait "$pid" 2>/dev/null
                local exit_code=$?
                if [[ $exit_code -eq 0 ]]; then
                    echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} Process for $model completed successfully"
                    wait_success=true
                else
                    echo -e "${RED}[$(date '+%H:%M:%S')]${NC} Process for $model failed (exit code: $exit_code)"
                    failed_models+=("$model")
                fi
                break
            fi
            
            # Check timeout
            local current_time=$(date +%s)
            if [[ $((current_time - wait_start)) -gt $timeout_seconds ]]; then
                echo -e "${RED}[$(date '+%H:%M:%S')]${NC} Timeout waiting for $model, killing process..."
                kill -TERM "$pid" 2>/dev/null
                sleep 2
                kill -KILL "$pid" 2>/dev/null
                failed_models+=("$model")
                break
            fi
            
            sleep 1
        done
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
        local results_file="evaluation_results_${model}.json"
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
