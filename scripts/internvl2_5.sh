#!/bin/bash
conda activate image_video

MODEL_NAME="internvl2_5"

GPU_ID="6"

HAS_IMAGE="1"                      
IMAGE_POS="after"                  
NFRAMES="32"                       
TARGET_RESOLUTION="(360, 420)"        
KEEP_ASPECT_RATIO="true"              

OUTPUT_FILE="${OUTPUT_DIR}/${MODEL_NAME}_${NFRAMES}.jsonl"
if [ "$HAS_IMAGE" = "0" ]; then
    OUTPUT_FILE="${OUTPUT_DIR}/${MODEL_NAME}_${HAS_IMAGE}_${NFRAMES}.jsonl"
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH=IV-Bench

python inference_ivbench.py \
    --model_name="$MODEL_NAME" \
    --question_file="$QUESTION_FILE" \
    --model_path="$MODEL_PATH" \
    --video_dir="$VIDEO_DIR" \
    --image_dir="$IMAGE_DIR" \
    --has_image="$HAS_IMAGE" \
    --nframes="$NFRAMES" \
    --output_file="$OUTPUT_FILE"
