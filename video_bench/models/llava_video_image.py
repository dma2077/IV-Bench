from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import requests
import copy
import torch
import sys
import warnings
from decord import VideoReader, cpu
import numpy as np
from video_bench.models.basic_model import BasicModel
from video_bench.registry import register_model


warnings.filterwarnings("ignore")
def load_video(video_path, max_frames_num,fps=1,force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # import pdb;pdb.set_trace()
    return spare_frames,frame_time,video_time

@register_model("llava_video_image")
class LLaVA_Video(BasicModel):
    def __init__(
        self,
        model_path: str = "lmms-lab/LLaVA-Video-7B-Qwen2",
        num_segments: int = 32
    ):
        super().__init__(model_path)
        self.max_frames = num_segments
        self.device = "cuda"
        model_name = "llava_qwen"
        
        self._tokenizer, self._model, self._image_processor, max_length = load_pretrained_model(
            model_path, 
            None, 
            model_name, 
            torch_dtype="bfloat16", 
            device_map="auto",
            overwrite_config = {'tie_word_embeddings': False, 'use_cache': True, "vocab_size": 151649}
            )
        self._model.eval()
    def set_frame_num(self, new_num):
        self.max_frames = new_num

    def generate_until(self, visual1: str, text: str) -> str:
        video_path = visual1
        video,frame_time,video_time = load_video(video_path, self.max_frames, 1, force_sample=True)
        video = self._image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
        video = [video]
        conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
        time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
        question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruciton}\n" + text
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt_question, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        cont = self._model.generate(
            input_ids,
            images=video,
            modalities= ["video"],
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
        )
        text_outputs = self._tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
        return text_outputs
    
    def generate_until1(self, visual1: str, visual2: str, text: str) -> str:
        video_path = visual1
        image_path = visual2
        video,frame_time,video_time = load_video(video_path, self.max_frames, 1, force_sample=True)
        frame_height, frame_width = video.shape[1:3]
        image = Image.open(image_path).convert("RGB")
        image = image.resize((frame_width, frame_height))
        image_array = np.array(image)

        if video.dtype != image_array.dtype:
            image_array = image_array.astype(video.dtype)
        image_array = np.expand_dims(image_array, axis=0)  # 从 (H, W, 3) 变为 (1, H, W, 3)

        print(f"Image array shape after expand_dims: {image_array.shape}")

        if video.shape[1:] != image_array.shape[1:]:
            raise ValueError(f"Shape mismatch: video shape {video.shape}, image shape {image_array.shape}")

        video = np.concatenate([video, image_array], axis=0)
        video = self._image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
        image = self._image_processor.preprocess(image_array, return_tensors="pt")["pixel_values"].cuda().bfloat16()
        video_image = [video, image]
        conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
        time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
        question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruciton}\n" + DEFAULT_IMAGE_TOKEN + text
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt_question, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        cont = self._model.generate(
            input_ids,
            images=video_image,
            modalities= ["video", "image"],
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
        )
        text_outputs = self._tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
        return text_outputs
