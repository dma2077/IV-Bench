# Image Understanding
CUDA_VISIBLE_DEVICES=0 python llavamini/eval/run_llava_mini.py \
    --model-path  ICTNLP/llava-mini-llama-3.1-8b \
    --image-file llavamini/serve/examples/baby_cake.png \
    --conv-mode llava_llama_3_1 --model-name "llava-mini" \
    --query "What's the text on the cake?"

# Video Understanding
CUDA_VISIBLE_DEVICES=0 python llavamini/eval/run_llava_mini.py \
    --model-path  ICTNLP/llava-mini-llama-3.1-8b \
    --video-file llavamini/serve/examples/fifa.mp4 \
    --conv-mode llava_llama_3_1 --model-name "llava-mini" \
    --query "What happened in this video?"