#!/bin/bash
LLAVA_MINI_ROOT=path_to_llama_mini_dir
gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

model_path=path_to_llava_mini_ckpt
CKPT=$(basename "$CKPT")
echo "Model path is set to: $model_path"

SPLIT="llava_gqa_testdev_balanced"
GQA_DATA=$LLAVA_MINI_ROOT/playground/data
GQADIR=$LLAVA_MINI_ROOT/playground/data/eval/gqa/data

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python $LLAVA_MINI_ROOT/llavamini/eval/model_vqa_loader.py \
        --model-path ${model_path} \
        --question-file $GQA_DATA/eval/gqa/$SPLIT.jsonl \
        --image-folder $GQA_DATA/eval/gqa/data/images \
        --answers-file ./playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode llava_llama_3_1 --model-name llava-mini &
done

wait

output_file=./playground/data/eval/gqa/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQA_DATA/eval/gqa/data/testdev_balanced_predictions.json

cd $GQA_DATA/eval/gqa/data
python eval/eval.py --tier testdev_balanced > ./playground/data/eval/gqa/answers/$SPLIT/$CKPT/res.txt

cat ./playground/data/eval/gqa/answers/$SPLIT/$CKPT/res.txt
