import argparse
import torch

from llamavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llamavid.conversation import conv_templates, SeparatorStyle
from llamavid.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

import json
import os
import math
from tqdm import tqdm
from decord import VideoReader, cpu
from PIL import Image
import numpy as np

from video_bench.models.basic_model import BasicModel
from video_bench.registry import register_model
from video_bench.res_smart import smart_resize_with_target

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def load_video(video_path, max_num_frames):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    
    num_frames_to_extract = min(max_num_frames, total_frame_num)
    
    frame_idx = [int(i * total_frame_num / num_frames_to_extract) for i in range(num_frames_to_extract)]
    
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames


def load_video_res(video_path, max_num_frames, target_resolution=None, keep_aspect_ratio = True, min_pixels = None, max_pixels = None):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    
    num_frames_to_extract = min(max_num_frames, total_frame_num)
    
    frame_idx = [int(i * total_frame_num / num_frames_to_extract) for i in range(num_frames_to_extract)]
    
    frames = vr.get_batch(frame_idx).asnumpy()
    
    if target_resolution is not None:
        resized_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            if min_pixels is not None and max_pixels is not None:
                img = smart_resize_with_target(img, target_resolution, keep_aspect_ratio, min_pixels=min_pixels, max_pixels=max_pixels)
            else:
                img = smart_resize_with_target(img, target_resolution, keep_aspect_ratio)
            resized_frames.append(np.array(img)) 
        frames = np.stack(resized_frames) 
    
    return frames

@register_model("llamavid")
class LLamaVID(BasicModel):
    def __init__(
        self, model_path: str="YanweiLi/llama-vid-7b-full-224-long-video",
    ):
        super().__init__(model_path)
        model_name = get_model_name_from_path(model_path)
        self._tokenizer, self._model, self._image_processor, self._context_len = load_pretrained_model(model_path, None, model_name)
        self.max_num_frames = 128

    def set_frame_num(self, new_num):
        self.max_num_frames = new_num
    
    
    def generate_until(self, visual, text) -> str:

        video_formats = ['.mp4', '.avi', '.mov', '.mkv']

        video_path = visual

        # Check if the video exists
        if os.path.exists(video_path):
            video = load_video(video_path, self.max_num_frames)
            video = self._image_processor.preprocess(video, return_tensors='pt')['pixel_values'].half().cuda()
            video = [video]

            qs = text
            if self._model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates['vicuna_v1'].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self._tokenizer, input_ids)

            cur_prompt = text
            with torch.inference_mode():
                self._model.update_prompt([[cur_prompt]])
                output_ids = self._model.generate(
                    input_ids,
                    images=video,
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = self._tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            return outputs



    def generate_until1(self, visual1, visual2, text) -> str:

        video_formats = ['.mp4', '.avi', '.mov', '.mkv']

        video_path = visual1
        image_path = visual2

        # Check if the video exists
        if os.path.exists(video_path) and os.path.exists(image_path):
            video = load_video(video_path, self.max_num_frames)
            image = Image.open(image_path).convert("RGB")
            image_resized = image.resize((video.shape[2], video.shape[1])) 
            image_np = np.array(image_resized)
            video_list = list(video)
            video_list.append(image_np) 
            video = np.array(video_list)
            video = self._image_processor.preprocess(video, return_tensors='pt')['pixel_values'].half().cuda()
            video = [video]

            qs = text
            if self._model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates['vicuna_v1'].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self._tokenizer, input_ids)

            cur_prompt = text
            with torch.inference_mode():
                self._model.update_prompt([[cur_prompt]])
                output_ids = self._model.generate(
                    input_ids,
                    images=video,
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = self._tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            return outputs


    def generate_until2(self, visual1, visual2, text, target_resolution = None, keep_aspect_ratio = True, min_pixels = None, max_pixels = None) -> str:

        video_path = visual1
        image_path = visual2

        # Check if the video exists
        if os.path.exists(video_path) and os.path.exists(image_path):
            
            video = load_video_res(video_path, self.max_num_frames, target_resolution, keep_aspect_ratio, min_pixels, max_pixels)
            image = Image.open(image_path).convert("RGB")
            image_resized = image.resize((video.shape[2], video.shape[1])) 
            image_np = np.array(image_resized) 
            video_list = list(video)
            video_list.append(image_np) 
            video = np.array(video_list)
            video = self._image_processor.preprocess(video, return_tensors='pt')['pixel_values'].half().cuda()
            video = [video]

            qs = text
            if self._model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates['vicuna_v1'].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self._tokenizer, input_ids)

            cur_prompt = text
            with torch.inference_mode():
                self._model.update_prompt([[cur_prompt]])
                output_ids = self._model.generate(
                    input_ids,
                    images=video,
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = self._tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            return outputs

    def generate_video_only(self, visual, text, nframes) -> str:
        video_path = visual

        if os.path.exists(video_path):
            video = load_video(video_path, self.max_num_frames)
            video = self._image_processor.preprocess(video, return_tensors='pt')['pixel_values'].half().cuda()
            video = [video]

            qs = text
            if self._model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates['vicuna_v1'].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stopping_criteria = KeywordsStoppingCriteria([stop_str], self._tokenizer, input_ids)

            cur_prompt = text
            with torch.inference_mode():
                self._model.update_prompt([[cur_prompt]])
                output_ids = self._model.generate(
                    input_ids,
                    images=video,
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria]
                )

            input_token_len = input_ids.shape[1]
            outputs = self._tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            return outputs.strip()
    def generate_video_only_res(self, visual, text, target_resolution=None, keep_aspect_ratio=True, min_pixels=None, max_pixels=None) -> str:
        video_path = visual

        if os.path.exists(video_path):
            video = load_video_res(video_path, self.max_num_frames, target_resolution, keep_aspect_ratio, min_pixels, max_pixels)
            video = self._image_processor.preprocess(video, return_tensors='pt')['pixel_values'].half().cuda()
            video = [video]

            qs = text
            if self._model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates['vicuna_v1'].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stopping_criteria = KeywordsStoppingCriteria([stop_str], self._tokenizer, input_ids)

            cur_prompt = text
            with torch.inference_mode():
                self._model.update_prompt([[cur_prompt]])
                output_ids = self._model.generate(
                    input_ids,
                    images=video,
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria]
                )

            input_token_len = input_ids.shape[1]
            outputs = self._tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            return outputs.strip()
