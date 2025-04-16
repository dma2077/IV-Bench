#!/bin/bash
LLAVA_MINI_ROOT=path_to_llama_mini_dir
VIDEO_BENCHMARK_ROOT=path_to_video_benchmarks_dir
model_path=path_to_llava_mini_ckpt

gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

CKPT=$(basename "$CKPT")
echo "Model path is set to: $model_path"

video_dir=$VIDEO_BENCHMARK_ROOT/egoschema/videos
gt_file=$VIDEO_BENCHMARK_ROOT/egoschema/mc.json
exp_dir=$LLAVA_MINI_ROOT/exp
output_dir=$exp_dir/egoschema/$CKPT

for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 $LLAVA_MINI_ROOT/llavamini/eval/video/egoschema_mc.py \
      --model_path ${model_path} \
      --video_dir ${video_dir} \
      --gt_file ${gt_file} \
      --output_dir ${output_dir} \
      --output_name ${CHUNKS}_${IDX} \
      --num_chunks $CHUNKS \
      --chunk_idx $IDX \
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