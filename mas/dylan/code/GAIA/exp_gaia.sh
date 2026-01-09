#!/bin/bash

MODEL=gpt-5-mini
# MODEL=gpt-4

# Specify your GAIA dataset directory
GAIA_DIR="/common/home/vv382/Datasets/GAIA"
EXP_NAME="gaia_experiment_iter_rm3"
OUTPUT_DIR="GAIA/results_gaia"

# Agent roles for GAIA
ROLES="['Assistant', 'Historian', 'Mathematician', 'Logician']"
# Process different file ranges or levels
# Example: Process first 50 problems (0-49)
MIN_FILE=0
MAX_FILE=102

RES_NAME="${OUTPUT_DIR}_Assistant_Historian_Mathematician_Logician/${EXP_NAME}_results.txt"
LOG_NAME="${OUTPUT_DIR}_Assistant_Historian_Mathematician_Logician/${EXP_NAME}_log.txt"

# Check if already processed (4 lines means complete)
if [ -f "$RES_NAME" ]; then
    linecount=$(wc -l < "$RES_NAME")
    if [ "$linecount" -eq 6 ]; then
        echo "Already processed: $RES_NAME"
        exit 0
    fi
fi

# Run GAIA experiment
echo "Running GAIA experiment from problem $MIN_FILE to $MAX_FILE..."
python -Xfaulthandler GAIA/llmlp_gen_gaia.py "$GAIA_DIR" "$MIN_FILE" "$MAX_FILE" "$EXP_NAME" "$MODEL" "$OUTPUT_DIR" "$ROLES" 2>&1 > "$LOG_NAME"

echo "GAIA experiment completed"
echo "Results saved to: $RES_NAME"
echo "Logs saved to: $LOG_NAME"
