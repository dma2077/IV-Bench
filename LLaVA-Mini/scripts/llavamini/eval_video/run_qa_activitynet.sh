#!/bin/bash
LLAVA_MINI_ROOT=path_to_llama_mini_dir
VIDEO_BENCHMARK_ROOT=path_to_video_benchmarks_dir
model_path=path_to_llava_mini_ckpt

gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

CKPT=$(basename "$CKPT")
echo "Model path is set to: $model_path"

cache_dir="./cache_dir"
video_dir="${VIDEO_BENCHMARK_ROOT}/ActivityNet_Zero_Shot_QA/test_videos"
gt_file_question="${VIDEO_BENCHMARK_ROOT}/ActivityNet_Zero_Shot_QA/test_q.json"
gt_file_answers="${VIDEO_BENCHMARK_ROOT}/ActivityNet_Zero_Shot_QA/test_a.json"
exp_dir=$LLAVA_MINI_ROOT/exp
output_dir="${exp_dir}/ActivityNet_Zero_Shot_QA/${CKPT_NAME}.1fps"

for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 $LLAVA_MINI_ROOT/llavamini/eval/video/run_inference_video_qa_act.py \
      --model_path ${model_path} \
      --cache_dir ${cache_dir} \
      --video_dir ${video_dir} \
      --gt_file_question ${gt_file_question} \
      --gt_file_answers ${gt_file_answers} \
      --output_dir ${output_dir} \
      --output_name ${CHUNKS}_${IDX} \
      --num_chunks $CHUNKS \
      --chunk_idx $IDX \
      --conv_mode llava_llama_3_1 \
      --model_name llava-mini &
done

wait

output_file=${output_dir}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${output_dir}/${CHUNKS}_${IDX}.json >> "$output_file"
done