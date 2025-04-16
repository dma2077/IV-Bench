# IV-Bench

[**📖 Arxiv Paper**]() | [**🤗 Paper**]() | [**🤗 IV-Bench Dataset**]()

This is the repo for the paper **IV-Bench: Towards Image-Grounded Video Understanding Evaluation**. In this project, we propose **IV-Bench**, the first comprehensive benchmark for evaluating **Image-Grounded Video Understanding**. IV-Bench comprises **967 videos** paired with **2,585 meticulously annotated image-text queries**, where the images, collected from external sources rather than extracted from the videos themselves, provide the essential context required to accurately answer the queries. The dataset spans **5 major categories** and covers **13 distinct tasks** (7 perception and 6 reasoning tasks), ensuring substantial diversity across various scenarios and task types. We evaluate **24 state-of-the-art MLLMs**, including the latest closed-source models (e.g., Gemini-2-Flash and Gemini-2-Pro) and open-source models (e.g., InternVL2.5 and Qwen2.5-VL series). Our analysis identifies key factors affecting performance, including inference patterns, frame numbers, and resolution. We also develop a synthetic data generation approach leveraging existing video QA datasets, though improvements remain limited, highlighting the quality and inherent difficulty of IV-Bench. We hope IV-Bench will effectively promote the development of MLLMs in **Image-Grounded Video Understanding**.


<div align="center">
<img src=./imgs/overview.png width=90% />
</div>

## Representative examples from IV-Bench
Each IV-Bench sample consists of a video paired with an image-text query. The correct answer is marked in green, with relevant video frames also highlighted in green.

<div align="center">
<img src=./imgs/examples.png width=90% />
</div>

## Comparion with other video benchmarks
Different from other video benchmarks that contain only text-only queries or image-unnecessary queries, IV-Bench is the first manually annotated benchmark explicitly designed to evaluate image-grounded video understanding, employing two rigorous rounds of quality checks to ensure images are essential for correctly answering every query.

<div align="center">
<img src=./imgs/comparion.png width=90% />
</div>


## How to use IV-Bench

### 1. Installation

We have provided the complete environment configuration required for evaluating the models in the paper. For detailed installation instructions and dependency settings, please refer to the [installation.md](installation.md) file.

### 2. Download dataset

#### 2.1 Download testdata from huggingface

Download all data without videos from here()

#### 2.2 Video download

Download videos using the script we provide [download_video.sh](download_video.sh)

### 3. Model Evaluation

```python
conda activate image_video
python inference_ivbench.py \
    --model_name="$MODEL_NAME" \
    --question_file="$QUESTION_FILE" \
    --model_path="$MODEL_PATH" \
    --video_dir="$VIDEO_DIR" \
    --image_dir="$IMAGE_DIR" \
    --has_image="$HAS_IMAGE" \
    --nframes="$NFRAMES" \
    --output_file="$OUTPUT_FILE"
```

One example for evaluating InternVL-2.5 can be seen in [internvl2_5.sh](scripts/internvl2_5.sh)

## Overall Leaderboard

<div align="center">
<img src=./imgs/experiment.png width=90% />
</div>


## Reference

```bib

```