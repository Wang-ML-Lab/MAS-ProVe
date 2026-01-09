#!/bin/bash

MODEL=gpt-5-mini

# Specify your AIME dataset directory and year
AIME_DIR="/common/home/vv382/Datasets/AIME/2024"
EXP_NAME="aime24_proc_prm3"
OUTPUT_DIR="AIME/results_aime"

# Agent roles for AIME (using math-specific experts)
ROLES="['Mathematician', 'AlgebraExpert', 'GeometryWizard', 'NumberTheoryScholar']"

# AIME has 30 problems (0-29)
# Control which problems to process
MIN_FILE=0
MAX_FILE=29

RES_NAME="${OUTPUT_DIR}_Mathematician_AlgebraExpert_GeometryWizard_NumberTheoryScholar/${EXP_NAME}.txt"
LOG_NAME="${OUTPUT_DIR}_Mathematician_AlgebraExpert_GeometryWizard_NumberTheoryScholar/${EXP_NAME}.log"

# Check if already processed (6 lines means complete)
if [ -f "$RES_NAME" ]; then
    linecount=$(wc -l < "$RES_NAME")
    if [ "$linecount" -eq 6 ]; then
        echo "Already processed: $RES_NAME"
        exit 0
    fi
fi

# Run AIME experiment
echo "Running AIME experiment from problem $MIN_FILE to $MAX_FILE..."
python AIME/llmlp_gen_aime_dylan.py "$AIME_DIR" "$MIN_FILE" "$MAX_FILE" "$EXP_NAME" "$MODEL" "$OUTPUT_DIR" "$ROLES" 2>&1 > "$LOG_NAME"

echo "AIME experiment completed"
echo "Results saved to: $RES_NAME"
echo "Logs saved to: $LOG_NAME"
