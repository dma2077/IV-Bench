# IV-Bench: A Benchmark for Image-Grounded Video Perception and Reasoning in Multimodal LLMs

[**🤗 IV-Bench Dataset**](https://huggingface.co/datasets/IV-Bench/IV-Bench)


IV‑Bench is a benchmark for evaluating the capabilities of multimodal large‑language models in image‑grounded video perception and reasoning. It pairs 966 videos with 2,560 externally sourced image–text queries, each requiring both video and image context for an accurate answer.


<div align="center">
<img src=imgs/overview.png width=90% />
</div>


## 👀 Instruction to IV-Bench

**IV-Bench** is the first comprehensive benchmark for evaluating **Image-Grounded Video perception and reasoning**. IV-Bench comprises **966 videos** paired with **2,560 meticulously annotated image-text queries**, where the images, collected from external sources rather than extracted from the videos themselves, provide the essential context required to accurately answer the queries. The dataset spans **5 major categories** and covers **13 distinct tasks** (7 perception and 6 reasoning tasks), ensuring substantial diversity across various scenarios and task types.

### Features

- **Image–Text Queries** Multiple queries per video, each pairing an externally sourced image with a question to provide essential contextual cues.

- **Five Diverse Categories** Videos (≥ 5 min) span Knowledge, Film & TV, Sports, Artistic Performances, and Life Records for broad coverage.

- **Thirteen Evaluation Tasks** A mix of perception and reasoning tasks designed to rigorously test multimodal understanding.


## 🎞️ Representative examples from IV-Bench
Each IV-Bench sample consists of a video paired with an image-text query. The correct answer is marked in green, with relevant video frames also highlighted in green.

<div align="center">
<img src=imgs/examples.png width=90% />
</div>

## 🆚 Comparion with other video benchmarks
Different from other video benchmarks that contain only text-only queries or image-unnecessary queries, IV-Bench is the first manually annotated benchmark explicitly designed to evaluate image-grounded video understanding, employing two rigorous rounds of quality checks to ensure images are essential for correctly answering every query.

<div align="center">
<img src=imgs/comparison.png width=90% />
</div>


## 🛠️ How to use IV-Bench

### 1. Installation

We have provided the complete environment configuration required for evaluating the models in the paper. For detailed installation instructions and dependency settings, please refer to the [installation.md](installation.md) file.

### 2. Download dataset from huggingface

huggingface-cli download --repo-type dataset --resume-download IV-Bench/IV-Bench --local-dir your_local_path

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

## 📊 Results

### Main Results

<div align="center">
<img src=imgs/experiment.png width=90% />
</div>

### Ablation Study

<div align="center">
<img src=imgs/ablation_study1.png width=90% />
</div>

<div align="center">
<img src=imgs/ablation_study2.png width=90% />
</div>

