# Environment Configuration

## Aria & InternVL-2/2.5 & LLaVA-OneVision & Minicpm-v & Phi3.5-vision & VideoLLaMA3

```bash


cd LLaVA-NeXT
conda create -n image_video python=3.10 -y

conda activate image_video

pip install --upgrade pip  # Enable PEP 660 support.

pip install -e ".[train]"

cd ..
conda env update --name image_video --file environment.yml
```

## Qwen2.5-VL
```bash
conda create -n qwen2_5 python=3.10 -y
conda activate qwen2_5
pip install git+https://github.com/huggingface/transformers accelerate
pip install qwen-vl-utils[decord]==0.0.10
```

## LLaMA-VID

```bash
git clone https://github.com/dvlab-research/LLaMA-VID.git

conda create -n llamavid python=3.10 -y
conda activate llamavid
cd LLaMA-VID
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install "numpy<2"
```

## Longva

```bash
git clone https://github.com/EvolvingLMMs-Lab/LongVA.git

conda create -n longva python=3.10 -y && conda activate longva
pip install torch==2.1.2 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -e "longva/.[train]"
pip install packaging &&  pip install ninja && pip install flash-attn==2.5.0 --no-build-isolation --no-cache-dir
pip install -r requirements.txt
```

## Phi4-multimodal

```bash
conda create -n phi4 python=3.10 -y && conda activate phi4
pip install torch==2.6.0
pip install flash_attn==2.7.4.post1 \
            transformers==4.48.2 \
            accelerate==1.3.0 \
            soundfile==0.13.1 \
            pillow==11.1.0 \
            scipy==1.15.2 \
            torchvision==0.21.0 \
            backoff==2.2.1 \
            peft==0.13.2 \
            tqdm==4.67.1 \
            opencv-python==4.11.0.86
```

## NVILA

```bash

conda create -n vila python=3.10 -y && conda activate vila
conda install -c nvidia cuda-toolkit -y
pip install --upgrade pip setuptools

pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

pip install -e ".[train,eval]"

pip install triton==3.1.0

site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')
cp -rv ./llava/train/deepspeed_replace/* $site_pkg_path/deepspeed/

pip install protobuf==3.20.*
```