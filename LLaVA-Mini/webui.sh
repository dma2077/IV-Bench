# Launch a controller
python -m llavamini.serve.controller --host 0.0.0.0 --port 10000 &

# Build the API of LLaVA-Mini
CUDA_VISIBLE_DEVICES=0  python -m llavamini.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path ICTNLP/llava-mini-llama-3.1-8b --model-name llava-mini &

# Start the interactive interface
python -m llavamini.serve.gradio_web_server --controller http://localhost:10000 --model-list-mode reload  --port 7860