#!/bin/bash

MODEL=gpt-5-mini
# MODEL=gpt-4

QUERY_DIR=/common/home/vv382/Datasets/GPQA/diamond
DIR_NAME=gpqa_diamond
EXP_NAME=gpqa_diamond_judge_iter

ROLES="['Physicist', 'Chemist', 'Biologist', 'Researcher']"

MIN_FILE=${MIN_FILE:-0}
MAX_FILE=${MAX_FILE:-197}
ASYNC_BATCH_SIZE=${ASYNC_BATCH_SIZE:-20}

# python llmlp_listwise_gpqa.py "$QUERY_DIR" "$MIN_FILE" "$MAX_FILE" "$EXP_NAME" "$MODEL" "$DIR_NAME" "$ROLES" "$ASYNC_BATCH_SIZE"
python llmlp_gen_gpqa_dylan.py "$QUERY_DIR" "$MIN_FILE" "$MAX_FILE" "$EXP_NAME" "$MODEL" "$DIR_NAME" "$ROLES" "$ASYNC_BATCH_SIZE"