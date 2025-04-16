# Evaluation

We evaluate LLaVA-Mini on 11 image benchmarks and 7 video benchmarks. Here, we provide the evaluation script. To make the evaluation process much faster, we provide the multi-GPU parallel version.

## Image-based Benchmarks

The evaluation pipelines for all image-based benchmarks are consistent with those used in [LLaVA-v1.5](https://github.com/haotian-liu/LLaVA). Before preparing task-specific data, **first download [eval.zip](https://drive.google.com/file/d/1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy/view?usp=sharing)**. It contains custom annotations, scripts, and the prediction files with LLaVA v1.5. Extract to `./playground/data/eval`. This also provides a general structure for all datasets.

### VQAv2

1. Download [`test2015`](http://images.cocodataset.org/zips/test2015.zip) and put it under `./playground/data/eval/vqav2`.
2. Inference.
```Shell
bash scripts/llavamini/eval_image/vqav2.sh
```
3. Submit the results to the [evaluation server](https://eval.ai/web/challenges/challenge-page/830/my-submission): `./playground/data/eval/vqav2/answers_upload`.

### GQA

1. Download the [data](https://cs.stanford.edu/people/dorarad/gqa/download.html) and [evaluation scripts](https://cs.stanford.edu/people/dorarad/gqa/evaluate.html) following the official instructions and put under `./playground/data/eval/gqa/data`. You may need to modify `eval.py` as [this](https://gist.github.com/haotian-liu/db6eddc2a984b4cbcc8a7f26fd523187) due to the missing assets in the GQA v1.2 release.
2. Inference and evaluate.
```Shell
bash scripts/llavamini/eval_image/gqa.sh
```

### VisWiz

1. Download [`test.json`](https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip) and extract [`test.zip`](https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip) to `test`. Put them under `./playground/data/eval/vizwiz`.
2. Inference.
```Shell
bash scripts/llavamini/eval_image/vizwiz.sh
```
3. Submit the results to the [evaluation server](https://eval.ai/web/challenges/challenge-page/2185/my-submission): `./playground/data/eval/vizwiz/answers_upload`.

### ScienceQA

1. Under `./playground/data/eval/scienceqa`, download `images`, `pid_splits.json`, `problems.json` from the `data/scienceqa` folder of the ScienceQA [repo](https://github.com/lupantech/ScienceQA).
2. Inference and evaluate.
```Shell
bash scripts/llavamini/eval_image/sqa.sh
```

### TextVQA

1. Download [`TextVQA_0.5.1_val.json`](https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json) and [images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip) and extract to `./playground/data/eval/textvqa`.
2. Inference and evaluate.
```Shell
bash scripts/llavamini/eval_image/textvqa.sh
```

### POPE

1. Download `coco` from [POPE](https://github.com/AoiDragon/POPE/tree/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco) and put under `./playground/data/eval/pope`.
2. Inference and evaluate.
```Shell
bash scripts/llavamini/eval_image/pope.sh
```

### MME

1. Download the data following the official instructions [here](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation).
2. Downloaded images to `MME_Benchmark_release_version`.
3. put the official `eval_tool` and `MME_Benchmark_release_version` under `./playground/data/eval/MME`.
4. Inference and evaluate.
```Shell
bash scripts/llavamini/eval_image/mme.sh
```

### MMBench

1. Download [`mmbench_dev_20230712.tsv`](https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_20230712.tsv) and put under `./playground/data/eval/mmbench`.
2. Inference.
```Shell
bash scripts/llavamini/eval_image/mmbench.sh
```
3. Submit the results to the [evaluation server](https://opencompass.org.cn/leaderboard-multimodal): `./playground/data/eval/mmbench/answers_upload/mmbench_dev_20230712`.

### MMBench-CN

1. Download [`mmbench_dev_cn_20231003.tsv`](https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_cn_20231003.tsv) and put under `./playground/data/eval/mmbench`.
2. Inference.
```Shell
bash scripts/llavamini/eval_image/mmbench_cn.sh
```
3. Submit the results to the evaluation server: `./playground/data/eval/mmbench/answers_upload/mmbench_dev_cn_20231003`.


### SEED-Bench

1. Following the official [instructions](https://github.com/AILab-CVC/SEED-Bench/blob/main/DATASET.md) to download the images and the videos. Put images under `./playground/data/eval/seed_bench/SEED-Bench-image`.
2. Extract the video frame in the middle from the downloaded videos, and put them under `./playground/data/eval/seed_bench/SEED-Bench-video-image`. We provide our script `extract_video_frames.py` modified from the official one.
3. Inference and evaluate.
```Shell
bash scripts/llavamini/eval_image/seed.sh
```
4. Optionally, submit the results to the leaderboard: `./playground/data/eval/seed_bench/answers_upload` using the official jupyter notebook.

### LLaVA-Bench-in-the-Wild

1. Extract contents of [`llava-bench-in-the-wild`](https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild) to `./playground/data/eval/llava-bench-in-the-wild`.
2. Inference and evaluate.
```Shell
bash scripts/llavamini/eval_image/llavabench.sh
```

### MM-Vet

1. Extract [`mm-vet.zip`](https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip) to `./playground/data/eval/mmvet`.
2. Inference.
```Shell
bash scripts/llavamini/eval_image/mmvet.sh
```
3. Evaluate the predictions in `./playground/data/eval/mmvet/results` using the official jupyter notebook.


## Video-baed Benchmarks

### Video-based generative performance benchmark

1. Download the videos from [here](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EatOpE7j68tLm2XAd0u6b8ABGGdVAwLMN6rqlDGM_DwhVA?e=90WIuW), and the question-answer pairs from [here](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EoS-mdm-KchDqCVbGv8v-9IB_ZZNXtcYAHtyvI06PqbF_A?e=1sNbaa).
2. Inference.
```Shell
bash scripts/llavamini/eval_video/run_general_benckmarking.sh
bash scripts/llavamini/eval_video/run_temporal_benckmarking.sh
bash scripts/llavamini/eval_video/run_consistency_benckmarking.sh
```
3. Evaluate using gpt-3.5-turbo.
```Shell
bash scripts/llavamini/eval_video/eval_benchmark_1_correctness.sh
bash scripts/llavamini/eval_video/eval_benchmark_2_detail.sh
bash scripts/llavamini/eval_video/eval_benchmark_3_contextual.sh
bash scripts/llavamini/eval_video/eval_benchmark_4_temporal.sh
bash scripts/llavamini/eval_video/eval_benchmark_5_consistency.sh
```

### MSVD-QA
1. Download video and question from [here](https://drive.google.com/file/d/1yXh9lz7flQ5Ui2IRSd6Qi6RqSEeUJwl3/view?pli=1).
2. Inference.
```Shell
bash scripts/llavamini/eval_video/run_qa_msvd.sh
```
3. Evaluate using gpt-3.5-turbo.
```Shell
bash scripts/llavamini/eval_video/eval_qa_msvd.sh
```

### MSVD-QA
1. Download video and question from [here](https://drive.google.com/file/d/1_q4eiSdb7i8P3Hmh4lCfgY1uBGyzU_7X/view).
2. Inference.
```Shell
bash scripts/llavamini/eval_video/run_qa_msvd.sh
```
3. Evaluate using gpt-3.5-turbo.
```Shell
bash scripts/llavamini/eval_video/eval_qa_msvd.sh
```

### MSRVTT-QA	
1. Download video and question from [here](https://drive.google.com/file/d/1yXh9lz7flQ5Ui2IRSd6Qi6RqSEeUJwl3/view).
2. Inference.
```Shell
bash scripts/llavamini/eval_video/run_qa_msrvtt.sh
```
3. Evaluate using gpt-3.5-turbo.
```Shell
bash scripts/llavamini/eval_video/eval_qa_msrvtt.sh
```

### Activitynet-QA	
1. Download video and question following [offical repo](https://github.com/MILVLG/activitynet-qa/tree/master/dataset).
2. Inference.
```Shell
bash scripts/llavamini/eval_video/run_qa_activitynet.sh
```
3. Evaluate using gpt-3.5-turbo.
```Shell
bash scripts/llavamini/eval_video/eval_qa_activitynet.sh
```

### MVBench
1. Download video and question following [offical repo](https://huggingface.co/datasets/OpenGVLab/MVBench).
2. Inference and evaluate.
```Shell
bash scripts/llavamini/eval_video/run_mvbench_mc.sh
```

### MLVU
1. Download video and question following [offical repo](https://github.com/JUNJIE99/MLVU).
2. Inference and evaluate.
```Shell
bash scripts/llavamini/eval_video/run_mlvu_mc.sh
```

### EgoSchema
1. Download video and question following [offical repo](https://github.com/egoschema/EgoSchema).
2. Inference and evaluate.
```Shell
bash scripts/llavamini/eval_video/run_egoschema_mc.sh
```
