#!/bin/bash

# This code might cause zombies. If you run it, you might have to kill the processes manually.

set -euo pipefail

# Ctrl+C handler: kill child processes from this script, then exit.
trap 'echo "Interrupted. Stopping child processes..."; pkill -P $$ || true; exit 130' INT TERM

MODEL=gpt-4.1-nano
# MODEL=gpt-4


# specify your directory
DIR_NAME=code_human_eval_judge_iter2_incontext_all_tests_t0.8

ROLES="['PythonAssistant', 'AlgorithmDeveloper', 'ComputerScientist', 'Programmer']"
JUDGES="['Passer', 'Tester', 'Reflector', 'Ranker']"

# 1 => quick sample run (single part, first few tasks), 0 => full run
SAMPLE_RUN=${SAMPLE_RUN:-1}
MAX_TASKS_PER_PART=${MAX_TASKS_PER_PART:-50}
QUIET=${QUIET:-1}
# Async batch size for llmlp_listwise_human_eval.py
ASYNC_BATCH_SIZE=${ASYNC_BATCH_SIZE:-30}

if [ "$SAMPLE_RUN" -eq 1 ]; then
    PARTS="0"
    RUN_ID=0
else
    PARTS="0 1 2 3"
    RUN_ID=0
fi

for part in $PARTS
do
    EXP_NAME="llmlp_human_eval_${RUN_ID}_${part}"
    # run sequentially so tqdm progress bar is readable and Ctrl+C works predictably
    python llmlp_gen_humaneval_dylan.py "$part" "$EXP_NAME" "$MODEL" "$DIR_NAME" "$ROLES" "$JUDGES" "$MAX_TASKS_PER_PART" "$QUIET" "$ASYNC_BATCH_SIZE"
done
echo "All done"


DIR_NAME=code_human_eval_judge_iter2_incontext_all_tests_t0.8_PythonAssistant_AlgorithmDeveloper_ComputerScientist_Programmer
for i in {0..0}
do

    : > ${DIR_NAME}/llmlp_human_eval_${i}.jsonl

    for part in $PARTS
    do
    EXP_NAME="llmlp_human_eval_${i}_${part}"
    if [ -f "${DIR_NAME}/${EXP_NAME}_443.jsonl" ]; then
        cat ${DIR_NAME}/${EXP_NAME}_443.jsonl >> ${DIR_NAME}/llmlp_human_eval_${i}.jsonl
    fi
    done
    if [ "$SAMPLE_RUN" -eq 1 ]; then
        python evaluate_subset_humaneval.py \
            ${DIR_NAME}/llmlp_human_eval_${i}.jsonl \
            --output ${DIR_NAME}/llmlp_human_eval_${i}.txt
    else
        evaluate_functional_correctness \
            ${DIR_NAME}/llmlp_human_eval_${i}.jsonl \
            > ${DIR_NAME}/llmlp_human_eval_${i}.txt
    fi

done
