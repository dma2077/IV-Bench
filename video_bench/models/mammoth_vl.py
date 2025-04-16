import warnings

warnings.filterwarnings("ignore")

import torch
import numpy as np
import copy
from PIL import Image
from decord import VideoReader, cpu

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.conversation import conv_templates
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
)

from video_bench.models.basic_model import BasicModel
from video_bench.registry import register_model
from video_bench.res_smart import smart_resize_with_target

def extract_frames(video_path, max_frames=32):

    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)

    frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    frames_array = vr.get_batch(frame_indices).asnumpy()  # (max_frames, H, W, C)

    frames = [Image.fromarray(frame) for frame in frames_array]
    return frames

def extract_frames_res(video_path, max_frames=32, target_resolution=None, keep_aspect_ratio = True, min_pixels = None, max_pixels = None):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)

    frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    frames_array = vr.get_batch(frame_indices).asnumpy()  # (max_frames, H, W, C)

    if target_resolution is not None:
        if min_pixels is not None and max_pixels is not None:
            frames = [smart_resize_with_target(Image.fromarray(frame), target_resolution, keep_aspect_ratio, min_pixels=min_pixels, max_pixels=max_pixels) for frame in frames_array]
        else:
            frames = [smart_resize_with_target(Image.fromarray(frame), target_resolution, keep_aspect_ratio) for frame in frames_array]
    else:
        frames = [Image.fromarray(frame) for frame in frames_array]
    
    return frames


@register_model("mammoth_vl")
class MAmmoTH_VL(BasicModel):
    def __init__(self, 
        model_path: str = "MAmmoTH-VL/MAmmoTH-VL-8B",    ):

        self.max_frames = 128
        super().__init__(model_path)
        model_name = "llava_qwen"
        self._device = "cuda"
        self._device_map = "auto"

        llava_model_args = {
            "multimodal": True,
            "attn_implementation": "sdpa",
        }

        self._tokenizer, self._model, self._image_processor, _ = load_pretrained_model(
            model_path,
            None,
            model_name,
            device_map=self._device_map,
            **llava_model_args,
        )
        self._model.eval()

    def set_frame_num(self, new_num):
        self.max_frames = new_num

    def generate_until(self, visual: str, text: str) -> str:

        frames = extract_frames(video_path=visual, max_frames=self.max_frames)

        image_tensor = (
            self._image_processor.preprocess(frames, return_tensors="pt")[
                "pixel_values"
            ]
            .half()
            .cuda()
        )

        conv_template = "qwen_2_5"
        question = f"{DEFAULT_IMAGE_TOKEN}\n{text}"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt_question, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self._device)
        )

        cont = self._model.generate(
            input_ids,
            images=[image_tensor],
            do_sample=False,
            temperature=0,
            top_p=1.0,
            max_new_tokens=4096,
            modalities=["video"],
        )

        text_output = self._tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        return text_output

    def generate_until1(self, visual1: str, visual2: str, text: str) -> str:

        frames = extract_frames(video_path=visual1, max_frames= self.max_frames)

        image = Image.open(visual2).convert("RGB")
        frames.append(image)

        image_tensor = (
            self._image_processor.preprocess(frames, return_tensors="pt")[
                "pixel_values"
            ]
            .half()
            .cuda()
        )

        conv_template = "qwen_2_5"
        question = f"{DEFAULT_IMAGE_TOKEN}\n{text}"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt_question, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self._device)
        )

        cont = self._model.generate(
            input_ids,
            images=[image_tensor],
            do_sample=False,
            temperature=0,
            top_p=1.0,
            max_new_tokens=4096,
            modalities=["video"], 
        )

        text_output = self._tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        return text_output


    def generate_until2(self, visual1: str, visual2: str, text: str, target_resolution=None, keep_aspect_ratio = True, min_pixels = None, max_pixels = None) -> str:

        frames = extract_frames_res(visual1, self.max_frames, target_resolution, keep_aspect_ratio, min_pixels, max_pixels)

        image = Image.open(visual2).convert("RGB")
        frames.append(image)

        image_tensor = (
            self._image_processor.preprocess(frames, return_tensors="pt")[
                "pixel_values"
            ]
            .half()
            .cuda()
        )

        conv_template = "qwen_2_5"
        question = f"{DEFAULT_IMAGE_TOKEN}\n{text}"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt_question, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self._device)
        )

        cont = self._model.generate(
            input_ids,
            images=[image_tensor],
            do_sample=False,
            temperature=0,
            top_p=1.0,
            max_new_tokens=4096,
            modalities=["video"], 
        )

        text_output = self._tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        return text_output
    