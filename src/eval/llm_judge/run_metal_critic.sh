#!/bin/bash
# Portions of this file are adapted from METAL, which declares the MIT License in its README.
# This version has been substantially modified for CSPO.
# Modifications Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="src/eval/output/CSPO/merged.jsonl" 
WORKING_HOME_DIR="src/eval/llm_judge/result/CSPO"

MODEL="gpt-4o"
SYSTEM="table"
MAX_ITER=0
NEED_REGIONS="True"
N_PROCESS=1
CUDA_DEVICES=3

RANGES=()
for ((start=0; start<4000; start+=1000)); do
  end=$((start+1000))
  RANGES+=("$start-$end")
done

for DATA_RANGE in "${RANGES[@]}"
do
    EXP_NAME=${MODEL}_${SYSTEM}_${DATA_RANGE}
    
    RESULT_DIR="$WORKING_HOME_DIR/result_${DATA_RANGE}.jsonl"
    AGENT_KWARGS="{\"cuda_device\": \"$CUDA_DEVICES\", \"system\": \"$SYSTEM\"}"
    WORKING_DIR=$WORKING_HOME_DIR/$EXP_NAME/
    LOG_FILE_PATH=$WORKING_DIR/$EXP_NAME.error

    mkdir -p $WORKING_DIR

    echo "Launching job for range $DATA_RANGE ..."
    
    screen -dmS $EXP_NAME bash -c "
        cd $WORKING_DIR;
        python3 -u $SCRIPT_DIR/MultiAgentSystem_critic.py \
        --result_dir $RESULT_DIR \
        --exp_name $EXP_NAME \
        --max_iter $MAX_ITER \
        --need_regions_correction $NEED_REGIONS \
        --model $MODEL \
        --data_range $DATA_RANGE \
        --dataset_dir $DATASET_DIR \
        --working_dir $WORKING_DIR \
        --n_process $N_PROCESS \
        --agent_kwargs '$AGENT_KWARGS' \
        > $LOG_FILE_PATH 2>&1"
done
