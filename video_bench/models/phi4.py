import cv2
from PIL import Image
import requests
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import numpy as np
from video_bench.models.basic_model import BasicModel
from video_bench.registry import register_model
from video_bench.res_smart import smart_resize_with_target

def extract_frames(video_path, target_fps=1, frames_upbound=128, num_segments=32):
    def uniform_sample(lst, n):
        gap = len(lst) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [lst[i] for i in idxs]
    cap = cv2.VideoCapture(video_path)
    frames = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    video_time = total_frames / original_fps

    fps_ratio = round(original_fps / target_fps)
    frame_indices = list(range(0, total_frames, fps_ratio))
    frame_times = [i / original_fps for i in frame_indices]

    if len(frame_indices) > frames_upbound:
        frame_indices = np.linspace(
            0, total_frames - 1, min(frames_upbound, num_segments), dtype=int
        ).tolist()
        frame_times = [i / original_fps for i in frame_indices]

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))

    frame_times_str = ",".join([f"{t:.2f}s" for t in frame_times])

    cap.release()
    return frames, video_time, frame_times_str, len(frame_indices)


@register_model("phi4")
class Phi4(BasicModel):
    def __init__(
        self,
        model_path: str = "microsoft/Phi-4-multimodal-instruct",
    ):
        super().__init__(model_path)
        self.num_segments = 128
        # Note: set _attn_implementation='eager' if you don't have flash_attn installed
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cuda",
            trust_remote_code=True,
            torch_dtype="auto",
            _attn_implementation="flash_attention_2",
        ).cuda()

        # for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
        self._processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, num_crops=4
        )


    def set_frame_num(self, new_num):
        self.num_segments = new_num

    def generate_until(self, visual, text) -> str:
        images = []
        placeholder = ""

        images, _, _, _ = extract_frames(
            video_path=visual, target_fps=1, frames_upbound=128, force_sample=False
        )

        # Note: if OOM, you might consider reduce number of frames in this example.
        for i in range(1, len(images) + 1):
            placeholder += f"<|image_{i}|>"

        messages = [
            {"role": "user", "content": placeholder + text},
        ]

        prompt = self._processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self._processor(prompt, images, return_tensors="pt").to("cuda:0")

        generation_args = {
            "max_new_tokens": 1024,
            "temperature": 0.0,
            "do_sample": False,
        }

        generate_ids = self._model.generate(
            **inputs,
            eos_token_id=self._processor.tokenizer.eos_token_id,
            **generation_args,
            num_logits_to_keep=0
        )

        # remove input tokens
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
        response = self._processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return response

    def generate_until1(self, visual1, visual2, text) -> str:
        placeholder = ""

        images, _, _, _ = extract_frames(
            video_path=visual1, target_fps=1, frames_upbound=128, num_segments=self.num_segments
        )
        images.append(Image.open(visual2))
        # Note: if OOM, you might consider reduce number of frames in this example.
        for i in range(1, len(images) + 1):
            placeholder += f"<|image_{i}|>"

        messages = [
            {"role": "user", "content": placeholder + text},
        ]

        prompt = self._processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self._processor(prompt, images, return_tensors="pt").to("cuda:0")

        generation_args = {
            "max_new_tokens": 1024,
            "temperature": 0.0,
            "do_sample": False,
        }

        generate_ids = self._model.generate(
            **inputs,
            eos_token_id=self._processor.tokenizer.eos_token_id,
            **generation_args,
            num_logits_to_keep=0
        )

        # remove input tokens
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
        response = self._processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return response


    def generate_until2(self, visual1, visual2, text, target_resolution=None, keep_aspect_ratio = True, min_pixels = None, max_pixels = None) -> str:
        placeholder = ""

        images, _, _, _ = extract_frames(
            video_path=visual1, target_fps=1, frames_upbound=128, num_segments=self.num_segments
        )

        if target_resolution is not None:
            resized_frames = []
            for frame in images:
                if min_pixels is not None and max_pixels is not None:
                    img = smart_resize_with_target(frame, target_resolution, keep_aspect_ratio, min_pixels, max_pixels)
                else:
                    img = smart_resize_with_target(frame, target_resolution, keep_aspect_ratio)
                resized_frames.append(img)  
            images = resized_frames
        
        images.append(Image.open(visual2))
        # Note: if OOM, you might consider reduce number of frames in this example.
        for i in range(1, len(images) + 1):
            placeholder += f"<|image_{i}|>"

        messages = [
            {"role": "user", "content": placeholder + text},
        ]

        prompt = self._processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self._processor(prompt, images, return_tensors="pt").to("cuda:0")

        generation_args = {
            "max_new_tokens": 1024,
            "temperature": 0.0,
            "do_sample": False,
        }

        generate_ids = self._model.generate(
            **inputs,
            eos_token_id=self._processor.tokenizer.eos_token_id,
            **generation_args,
            num_logits_to_keep=0
        )

        # remove input tokens
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
        response = self._processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return response

    def generate_video_only(self, visual1, text, nframes) -> str:
        placeholder = ""

        images, _, _, _ = extract_frames(
            video_path=visual1, target_fps=1, frames_upbound=128, num_segments=self.num_segments
        )

        for i in range(1, len(images) + 1):
            placeholder += f"<|image_{i}|>"

        messages = [
            {"role": "user", "content": placeholder + text},
        ]

        prompt = self._processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self._processor(prompt, images, return_tensors="pt").to("cuda:0")

        generation_args = {
            "max_new_tokens": 1024,
            "temperature": 0.0,
            "do_sample": False,
        }

        generate_ids = self._model.generate(
            **inputs,
            eos_token_id=self._processor.tokenizer.eos_token_id,
            **generation_args,
            num_logits_to_keep=0
        )

        generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
        response = self._processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return response
    def generate_video_only_res(self, visual1, text, target_resolution=None, keep_aspect_ratio=True, min_pixels=None, max_pixels=None) -> str:
        placeholder = ""

        images, _, _, _ = extract_frames(
            video_path=visual1, target_fps=1, frames_upbound=128, num_segments=self.num_segments
        )

        if target_resolution is not None:
            resized_frames = []
            for frame in images:
                if min_pixels is not None and max_pixels is not None:
                    img = smart_resize_with_target(frame, target_resolution, keep_aspect_ratio, min_pixels, max_pixels)
                else:
                    img = smart_resize_with_target(frame, target_resolution, keep_aspect_ratio)
                resized_frames.append(img)
            images = resized_frames

        for i in range(1, len(images) + 1):
            placeholder += f"<|image_{i}|>"

        messages = [
            {"role": "user", "content": placeholder + text},
        ]

        prompt = self._processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self._processor(prompt, images, return_tensors="pt").to("cuda:0")

        generation_args = {
            "max_new_tokens": 1024,
            "temperature": 0.0,
            "do_sample": False,
        }

        generate_ids = self._model.generate(
            **inputs,
            eos_token_id=self._processor.tokenizer.eos_token_id,
            **generation_args,
            num_logits_to_keep=0
        )

        generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
        response = self._processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return response
