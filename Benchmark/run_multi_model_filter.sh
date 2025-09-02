#!/bin/bash

# Multi-model voting consensus filtering script
# Uses qwen2.5-32b-instruct and qwen-plus for voting

echo "=== Multi-Model Voting Consensus Filtering ==="
echo "Models: qwen2.5-32b-instruct + qwen-plus"
echo ""

# Input file
INPUT_FILE="sample_belief_update_zoning.jsonl"

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file $INPUT_FILE not found!"
    exit 1
fi

echo "Processing file: $INPUT_FILE"
echo ""

# Filter 1: Both models show context improvement (全部提升)
echo "=== Filter 1: All models show context improvement ==="
python filter_jsonl.py "$INPUT_FILE" \
    --models qwen2.5-32b-instruct qwen-plus \
    --filters all_models_context_improves \
    --max-workers 2 \
    --temperature 0

echo ""

# Filter 2: Both models have both predictions wrong (全错)
echo "=== Filter 2: All models have both predictions wrong ==="
python filter_jsonl.py "$INPUT_FILE" \
    --models qwen2.5-32b-instruct qwen-plus \
    --filters all_models_both_wrong \
    --max-workers 2 \
    --temperature 0

echo ""

# Filter 3: Both models show context degradation
echo "=== Filter 3: All models show context degradation ==="
python filter_jsonl.py "$INPUT_FILE" \
    --models qwen2.5-32b-instruct qwen-plus \
    --filters all_models_context_degrades \
    --max-workers 2 \
    --temperature 0

echo ""

# Filter 4: Both models have both predictions correct
echo "=== Filter 4: All models have both predictions correct ==="
python filter_jsonl.py "$INPUT_FILE" \
    --models qwen2.5-32b-instruct qwen-plus \
    --filters all_models_both_correct \
    --max-workers 2 \
    --temperature 0

echo ""

# Combination filter: Context improves OR both wrong
echo "=== Filter 5: All models context improves OR all models both wrong ==="
python filter_jsonl.py "$INPUT_FILE" \
    --models qwen2.5-32b-instruct qwen-plus \
    --filters all_models_context_improves all_models_both_wrong \
    --max-workers 2 \
    --temperature 0

echo ""
echo "=== Filtering completed! ==="
echo "Check the generated *_filtered_*_voting_*.jsonl files for results."
