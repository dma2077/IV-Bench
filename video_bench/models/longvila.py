import argparse
import csv
import itertools
import json
import os

import torch
from pygments.lexer import default
from tqdm import tqdm

import llava
from llava import conversation as conversation_lib
from llava.eval.mmmu_utils.eval_utils import parse_choice
from llava.utils import io
from llava.utils.logging import logger

from video_bench.models.basic_model import BasicModel
from video_bench.registry import register_model
from PIL import Image

from video_bench.models.basic_model import BasicModel
from video_bench.registry import register_model
from video_bench.res_smart import smart_resize_with_target

@register_model("longvila")
class LongVILA(BasicModel):
    def __init__(
        self, model_path: str="Efficient-Large-Model/qwen2-7b-longvila-1M",
    ):
        super().__init__(model_path)
        device = "cuda"
        device_map = "auto"
        self._model = llava.load(model_path, model_base = "llava_qwen", device_map=device_map)
        self.num_video_frames = 128
        if self.num_video_frames > 0:
            self._model.config.num_video_frames = self.num_video_frames

    def set_frame_num(self, new_num):
        self.num_video_frames = new_num
        self._model.config.num_video_frames = self.num_video_frames

    def generate_until(self, visual, text) -> str:
        generation_config_re = {"max_new_tokens": 1024, "do_sample": False}
        generation_config = self._model.default_generation_config
        if generation_config_re is not None:
            generation_config.update(**generation_config_re)

        video = llava.Video(visual)

        question = text

        response = self._model.generate_content([video, question], generation_config=generation_config)

        return response.replace("</s>", "").replace("</s", "").replace("<s>", "").replace("<s", "").replace("</", "")
    
    def generate_until1(self, visual1, visual2, text) -> str:
        generation_config_re = {"max_new_tokens": 1024, "do_sample": False}
        generation_config = self._model.default_generation_config
        if generation_config_re is not None:
            generation_config.update(**generation_config_re)

        video = llava.Video(visual1)

        # print(video.pathat)

        image = Image.open(visual2)

        question = text

        response = self._model.generate_content([video, image, question], generation_config=generation_config)

        return response.replace("</s>", "").replace("</s", "").replace("<s>", "").replace("<s", "").replace("</", "")
    
    def generate_until2(self, visual1, visual2, text, target_resolution=None, keep_aspect_ratio = True, min_pixels = None, max_pixels = None) -> str:
        generation_config_re = {"max_new_tokens": 1024, "do_sample": False}
        generation_config = self._model.default_generation_config
        if generation_config_re is not None:
            generation_config.update(**generation_config_re)

        video = llava.Video(visual1)

        image = Image.open(visual2)

        question = text

        response = self._model.generate_content([video, image, question], generation_config=generation_config)

        return response.replace("</s>", "").replace("</s", "").replace("<s>", "").replace("<s", "")
    
    def generate_video_only(self, visual1, text,nframes) -> str:
        generation_config_re = {"max_new_tokens": 1024, "do_sample": False}
        generation_config = self._model.default_generation_config
        if generation_config_re is not None:
            generation_config.update(**generation_config_re)

        video = llava.Video(visual1)

        question = text

        response = self._model.generate_content([video, question], generation_config=generation_config)

        return response.replace("</s>", "").replace("</s", "").replace("<s>", "").replace("<s", "").replace("</", "")
    def generate_video_only_res(self, visual1, text, target_resolution=None, keep_aspect_ratio=True, min_pixels=None, max_pixels=None) -> str:
        generation_config_re = {"max_new_tokens": 1024, "do_sample": False}
        generation_config = self._model.default_generation_config
        if generation_config_re is not None:
            generation_config.update(**generation_config_re)

        video = llava.Video(visual1)

        question = text

        response = self._model.generate_content([video, question], generation_config=generation_config)

        return response.replace("</s>", "").replace("</s", "").replace("<s>", "").replace("<s", "").replace("</", "")
