#!/bin/bash
LLAVA_MINI_ROOT=path_to_llama_mini_dir
model_path=path_to_llava_mini_ckpt

gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

CKPT=$(basename "$CKPT")
echo "Model path is set to: $model_path"

exp_dir=$LLAVA_MINI_ROOT/exp

pred_path="${exp_dir}/MSRVTT_Zero_Shot_QA/${CKPT_NAME}/merge.jsonl"
output_dir="${exp_dir}/MSRVTT_Zero_Shot_QA/${CKPT_NAME}/gpt-3.5-turbo"
output_json="${exp_dir}/MSRVTT_Zero_Shot_QA/${CKPT_NAME}/gpt-3.5-turbo_results.json"
api_key="XXXXXXXXXXXXXXXXXXXXXXXXXXXX"
api_base="https://api.openai.com/v1"
num_tasks=32

python3 $LLAVA_MINI_ROOT/llavamini/eval/video/eval_video_qa.py \
    --pred_path ${pred_path} \
    --output_dir ${output_dir} \
    --output_json ${output_json} \
    --api_key ${api_key} \
    --api_base ${api_base} \
    --model_name "gpt-3.5-turbo" \
    --num_tasks ${num_tasks} 
