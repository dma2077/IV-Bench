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


@register_model("nvila")
class NVILA(BasicModel):
    def __init__(
        self, model_path: str="Efficient-Large-Model/NVILA-8B",
    ):
        super().__init__(model_path)
        device = "cuda"
        device_map = "auto"
        self._model = llava.load(model_path, model_base = "llava_qwen", device_map=device_map)
        num_video_frames = 32
        if num_video_frames > 0:
            self._model.config.num_video_frames = num_video_frames

    def set_frame_num(self, new_num):
        self.max_num_frames = new_num

    def generate_until(self, visual, text) -> str:
        generation_config_re = {"max_new_tokens": 1024, "do_sample": False}
        generation_config = self._model.default_generation_config
        if generation_config_re is not None:
            generation_config.update(**generation_config_re)

        video = llava.Video(visual)

        question = text

        response = self._model.generate_content([video, question], generation_config=generation_config)

        return response

    def generate_until1(self, visual1, visual2, text) -> str:
        generation_config_re = {"max_new_tokens": 1024, "do_sample": False}
        generation_config = self._model.default_generation_config
        if generation_config_re is not None:
            generation_config.update(**generation_config_re)

        video = llava.Video(visual1)
        image = llava.Image(visual2)

        question = text

        response = self._model.generate_content([video, image, question], generation_config=generation_config)

        return response