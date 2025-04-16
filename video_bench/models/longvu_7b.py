import sys
import os
sys.path.append("IV-Bench/LongVU")


import numpy as np
import torch
from LongVU.longvu.builder import load_pretrained_model
from LongVU.longvu.constants import (
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from LongVU.longvu.conversation import conv_templates, SeparatorStyle
from LongVU.longvu.mm_datautils import (
    KeywordsStoppingCriteria,
    process_images,
    tokenizer_image_token,
)
from decord import cpu, VideoReader
from PIL import Image

from video_bench.models.basic_model import BasicModel
from video_bench.registry import register_model
from video_bench.res_smart import smart_resize_with_target

@register_model("longvu")
class LongVU(BasicModel):
    def __init__(
        self, model_path: str="Vision-CAIR/LongVU_Qwen2_7B",
    ):
        super().__init__(model_path)
        self._tokenizer, self._model, self._image_processor, self._context_len = load_pretrained_model(
            model_path, None, "cambrian_qwen",
        )
        self.max_num_frames = 128

        self._model.eval()
    
    def set_frame_num(self, new_num):
        self.max_num_frames = new_num

    def generate_until(self, visual, text) -> str:
        vr = VideoReader(visual, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)

        num_frames_to_extract = min(self.max_num_frames, total_frame_num)

        frame_indices = np.array([int(i * total_frame_num / num_frames_to_extract) for i in range(num_frames_to_extract)])

        video = []
        for frame_index in frame_indices:
            img = vr[frame_index].asnumpy()
            video.append(img)
            
        video = np.stack(video)
        image_sizes = [video[0].shape[:2]]
        video = process_images(video, self._image_processor, self._model.config)
        video = [item.unsqueeze(0) for item in video]

        qs = DEFAULT_IMAGE_TOKEN + "\n" + text
        conv = conv_templates["qwen"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self._model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self._tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self._model.generate(
                input_ids,
                images=video,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0.2,
                max_new_tokens=512,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )
        pred = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        return pred


    def generate_until1(self, visual1, visual2, text, nframes) -> str:

        vr = VideoReader(visual1, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)

        num_frames_to_extract = min(self.max_num_frames, total_frame_num)

        frame_indices = np.array([int(i * total_frame_num / num_frames_to_extract) for i in range(num_frames_to_extract)])

        video = []
        for frame_index in frame_indices:
            img = vr[frame_index].asnumpy()
            video.append(img)
        
        image = np.array(Image.open(visual2).convert("RGB"))
        if image.shape[:2] != video[0].shape[:2]:  
            image = np.array(Image.fromarray(image).resize((video[0].shape[1], video[0].shape[0])))
        video.append(image)

        video = np.stack(video)
        image_sizes = [video[0].shape[:2]]
        video = process_images(video, self._image_processor, self._model.config)
        video = [item.unsqueeze(0) for item in video]

        qs = DEFAULT_IMAGE_TOKEN + "\n" + text
        conv = conv_templates["qwen"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self._model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self._tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self._model.generate(
                input_ids,
                images=video,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0.2,
                max_new_tokens=512,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )
        pred = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        return pred


    def generate_until2(self, visual1, visual2, text, target_resolution=None, keep_aspect_ratio = True, min_pixels = None, max_pixels = None) -> str:

        vr = VideoReader(visual1, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)

        num_frames_to_extract = min(self.max_num_frames, total_frame_num)

        frame_indices = np.array([int(i * total_frame_num / num_frames_to_extract) for i in range(num_frames_to_extract)])

        video = []
        for frame_index in frame_indices:
            img = vr[frame_index].asnumpy()
            video.append(img)
        

        if target_resolution is not None:
            resized_frames = []
            for frame in video:
                img = Image.fromarray(frame)  
                img = smart_resize_with_target(img, target_resolution, keep_aspect_ratio, min_pixels, max_pixels)
                resized_frames.append(np.array(img)) 
            
            video = resized_frames
        
        image = np.array(Image.open(visual2).convert("RGB"))
        if image.shape[:2] != video[0].shape[:2]: 
            image = np.array(Image.fromarray(image).resize((video[0].shape[1], video[0].shape[0])))
        video.append(image)

        video = np.stack(video)
        image_sizes = [video[0].shape[:2]]
        video = process_images(video, self._image_processor, self._model.config)
        video = [item.unsqueeze(0) for item in video]

        qs = DEFAULT_IMAGE_TOKEN + "\n" + text
        conv = conv_templates["qwen"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self._model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self._tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self._model.generate(
                input_ids,
                images=video,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0.2,
                max_new_tokens=512,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )
        pred = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        return pred
    
    def generate_video_only(self, visual, text, nframes) -> str:
        vr = VideoReader(visual, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)

        num_frames_to_extract = min(self.max_num_frames, total_frame_num)

        frame_indices = np.array([int(i * total_frame_num / num_frames_to_extract) for i in range(num_frames_to_extract)])

        video = []
        for frame_index in frame_indices:
            img = vr[frame_index].asnumpy()
            video.append(img)

        video = np.stack(video)
        image_sizes = [video[0].shape[:2]]
        video = process_images(video, self._image_processor, self._model.config)
        video = [item.unsqueeze(0) for item in video]

        qs = DEFAULT_IMAGE_TOKEN + "\n" + text
        conv = conv_templates["qwen"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self._model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self._tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self._model.generate(
                input_ids,
                images=video,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0.2,
                max_new_tokens=512,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )
        pred = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return pred


    def generate_video_only_res(self, visual, text, target_resolution=None, keep_aspect_ratio=True, min_pixels=None, max_pixels=None) -> str:
        vr = VideoReader(visual, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)

        num_frames_to_extract = min(self.max_num_frames, total_frame_num)

        frame_indices = np.array([
            int(i * total_frame_num / num_frames_to_extract) 
            for i in range(num_frames_to_extract)
        ])

        video = []
        for frame_index in frame_indices:
            img = vr[frame_index].asnumpy()
            video.append(img)

        if target_resolution is not None:
            resized_frames = []
            for frame in video:
                img = Image.fromarray(frame)
                img = smart_resize_with_target(img, target_resolution, keep_aspect_ratio, min_pixels, max_pixels)
                resized_frames.append(np.array(img))
            video = resized_frames

        video = np.stack(video)
        image_sizes = [video[0].shape[:2]]
        video = process_images(video, self._image_processor, self._model.config)
        video = [item.unsqueeze(0) for item in video]

        qs = DEFAULT_IMAGE_TOKEN + "\n" + text
        conv = conv_templates["qwen"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self._model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self._tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self._model.generate(
                input_ids,
                images=video,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0.2,
                max_new_tokens=512,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        pred = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return pred
