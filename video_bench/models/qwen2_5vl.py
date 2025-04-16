from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import torch

from typing import List
from qwen_vl_utils import process_vision_info, extract_vision_info
from qwen_vl_utils.vision_process import smart_nframes
from accelerate import Accelerator, DistributedType
from video_bench.models.basic_model import BasicModel
from video_bench.registry import register_model
import os
import torch



@register_model("qwen2_5vl")
class Qwen2_5VL(BasicModel):
    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    ):
        super().__init__(model_path)

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
        ).eval()
        self._processor = AutoProcessor.from_pretrained(model_path)
        self._config = self._model.config
        self.max_num_frames = 128


    def set_frame_num(self, new_num):
        self.max_num_frames = new_num
        print(f"set max frames:{self.max_num_frames}!!!")

    def _process_inputs(self, visual, text: str):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": visual,
                        "nframes": self.max_num_frames,
                        "max_pixels": 360 * 420,
                        # "fps": 1.0,
                    },
                    {"type": "text", "text": text},
                ],
            }
        ]
        prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        inputs = self._processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to("cuda")

        return inputs

    def generate_until(self, visual, text) -> str:
        # Inference
        inputs = self._process_inputs(visual, text)
        generated_ids = self._model.generate(
            **inputs, max_new_tokens=1024, do_sample=False
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_text[0]

    def _process_inputs1(self, visual1, visual2, text: str):
        # Check if visual1, visual2, and text are all strings
        if isinstance(visual1, str) and isinstance(visual2, str) and isinstance(text, str):
            # Process single input
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": visual1,
                            "nframes": self.max_num_frames,
                            "max_pixels": 576 * 324,
                            # "fps": 1.0,
                        },
                        {
                            "type": "image", 
                            "image": visual2, 
                            # "max_pixels": 360 * 420
                         },
                        {"type": "text", "text": text},
                    ],
                }
            ]
            prompt = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            inputs = self._processor(
                text=[prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            )
            inputs = inputs.to("cuda")
        elif isinstance(visual1, list) and isinstance(visual2, list) and isinstance(text, list):
            if len(visual1) == len(visual2) == len(text):
                # Perform batch inference
                batch_messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": v1,
                                "nframes": self.max_num_frames,
                                # "max_pixels": 240 * 320,
                                # "fps": 1.0,
                            },
                            {"type": "image", 
                             "image": v2, 
                            #  "max_pixels": 360 * 420
                             },
                            {"type": "text", "text": t},
                        ],
                    }
                    for v1, v2, t in zip(visual1, visual2, text)
                ]
                batch_prompt = [
                    self._processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    for messages in batch_messages
                ]
                image_inputs, video_inputs, video_kwargs = process_vision_info(batch_messages, return_video_kwargs=True)
                print(video_inputs)
                inputs = self._processor(
                    text=batch_prompt,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                    **video_kwargs,
                )
                inputs = inputs.to("cuda")
            else:
                raise ValueError("The lists visual1, visual2, and text must have the same length for batch processing.")

        else:
            raise ValueError("visual1, visual2, and text must either all be strings or all be lists of the same length.")

        return inputs

    def generate_until1(self, visual1, visual2, text) -> str:
        # Inference
        inputs = self._process_inputs1(visual1, visual2, text)
        generated_ids = self._model.generate(
            **inputs, max_new_tokens=1024, do_sample=False
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_text[0]

    def _process_inputs2(self, visual1, visual2, text: str, target_resolution=None, keep_aspect_ratio = True, min_pixels = None, max_pixels = None):
        if target_resolution is not None:
            target_width, target_height = target_resolution
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": visual1,
                            "nframes": self.max_num_frames,
                            # "max_pixels": max_pixels,
                            # "min_pixels": min_pixels,
                            "resized_width": target_width,
                            "resized_height": target_height,
                            # "fps": 1.0,
                        },
                        {"type": "image", "image": visual2},
                        {"type": "text", "text": text},
                    ],
                }
            ]
            prompt = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self._processor(
                text=[prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            return inputs
        elif min_pixels is not None and max_pixels is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": visual1,
                            "nframes": self.max_num_frames,
                            "max_pixels": max_pixels,
                            "min_pixels": min_pixels,
                            # "resized_width": target_width,
                            # "resized_height": target_height,
                            # "fps": 1.0,
                        },
                        {"type": "image", "image": visual2},
                        {"type": "text", "text": text},
                    ],
                }
            ]
            prompt = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self._processor(
                text=[prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            return inputs
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": visual1,
                            "nframes": self.max_num_frames,
                            # "max_pixels": max_pixels,
                            # "min_pixels": min_pixels,
                            # "resized_width": target_width,
                            # "resized_height": target_height,
                            # "fps": 1.0,
                        },
                        {"type": "image", "image": visual2},
                        {"type": "text", "text": text},
                    ],
                }
            ]
            prompt = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self._processor(
                text=[prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            return inputs


    def generate_until2(self, visual1, visual2, text, target_resolution=None, keep_aspect_ratio = True, min_pixels = None, max_pixels = None) -> str:
        # Inference
        inputs = self._process_inputs2(visual1, visual2, text, target_resolution, keep_aspect_ratio, min_pixels, max_pixels)
        generated_ids = self._model.generate(
            **inputs, max_new_tokens=1024, do_sample=False
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_text[0]

    def _process_inputs3(self, visual1, visual2, text: str, total_pixels = None):
        if total_pixels is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": visual1,
                            "nframes": self.max_num_frames,
                            "total_pixels": total_pixels,
                            # "max_pixels": 360 * 420,
                            # "fps": 1.0,
                        },
                        {"type": "image", "image": visual2},
                        {"type": "text", "text": text},
                    ],
                }
            ]
            prompt = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self._processor(
                text=[prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            return inputs
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": visual1,
                            "nframes": self.max_num_frames,
                            # "max_pixels": 360 * 420,
                            # "fps": 1.0,
                        },
                        {"type": "image", "image": visual2},
                        {"type": "text", "text": text},
                    ],
                }
            ]
            prompt = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self._processor(
                text=[prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            return inputs

    def generate_until3(self, visual1, visual2, text, total_pixels = None) -> str:
        # Inference
        inputs = self._process_inputs3(visual1, visual2, text, total_pixels)
        generated_ids = self._model.generate(
            **inputs, max_new_tokens=1024, do_sample=False
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_text[0]