import os
import requests
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from decord import VideoReader
from PIL import Image
from tqdm import tqdm
from typing import List

from video_bench.models.basic_model import BasicModel
from video_bench.registry import register_model
from video_bench.res_smart import smart_resize_with_target

def load_video(
    video_file, num_frames=128, cache_dir="cached_video_frames", verbosity="DEBUG"
):
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    video_basename = os.path.basename(video_file)
    cache_subdir = os.path.join(cache_dir, f"{video_basename}_{num_frames}")
    os.makedirs(cache_subdir, exist_ok=True)

    cached_frames = []
    missing_frames = []
    frame_indices = []

    for i in range(num_frames):
        frame_path = os.path.join(cache_subdir, f"frame_{i}.jpg")
        if os.path.exists(frame_path):
            cached_frames.append(frame_path)
        else:
            missing_frames.append(i)
            frame_indices.append(i)

    vr = VideoReader(video_file)
    duration = len(vr)
    fps = vr.get_avg_fps()

    frame_timestamps = [
        int(duration / num_frames * (i + 0.5)) / fps for i in range(num_frames)
    ]

    if verbosity == "DEBUG":
        print(
            "Already cached {}/{} frames for video {}, enjoy speed!".format(
                len(cached_frames), num_frames, video_file
            )
        )
    # If all frames are cached, load them directly
    if not missing_frames:
        return [
            Image.open(frame_path).convert("RGB") for frame_path in cached_frames
        ], frame_timestamps

    actual_frame_indices = [
        int(duration / num_frames * (i + 0.5)) for i in missing_frames
    ]

    missing_frames_data = vr.get_batch(actual_frame_indices).asnumpy()

    for idx, frame_index in enumerate(tqdm(missing_frames, desc="Caching rest frames")):
        img = Image.fromarray(missing_frames_data[idx]).convert("RGB")
        frame_path = os.path.join(cache_subdir, f"frame_{frame_index}.jpg")
        img.save(frame_path)
        cached_frames.append(frame_path)

    cached_frames.sort(
        key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0])
    )
    return [
        Image.open(frame_path).convert("RGB") for frame_path in cached_frames
    ], frame_timestamps


def create_image_gallery(images, columns=3, spacing=20, bg_color=(200, 200, 200)):
    """
    Combine multiple images into a single larger image in a grid format.

    Parameters:
        image_paths (list of str): List of file paths to the images to display.
        columns (int): Number of columns in the gallery.
        spacing (int): Space (in pixels) between the images in the gallery.
        bg_color (tuple): Background color of the gallery (R, G, B).

    Returns:
        PIL.Image: A single combined image.
    """
    # Open all images and get their sizes
    img_width, img_height = images[0].size  # Assuming all images are of the same size

    # Calculate rows needed for the gallery
    rows = (len(images) + columns - 1) // columns

    # Calculate the size of the final gallery image
    gallery_width = columns * img_width + (columns - 1) * spacing
    gallery_height = rows * img_height + (rows - 1) * spacing

    # Create a new image with the calculated size and background color
    gallery_image = Image.new("RGB", (gallery_width, gallery_height), bg_color)

    # Paste each image into the gallery
    for index, img in enumerate(images):
        row = index // columns
        col = index % columns

        x = col * (img_width + spacing)
        y = row * (img_height + spacing)

        gallery_image.paste(img, (x, y))

    return gallery_image


def get_placeholders_for_videos(frames: List, timestamps=[]):
    contents = []
    if not timestamps:
        for i, _ in enumerate(frames):
            contents.append({"text": None, "type": "image"})
        contents.append({"text": "\n", "type": "text"})
    else:
        for i, (_, ts) in enumerate(zip(frames, timestamps)):
            contents.extend(
                [
                    {"text": f"[{int(ts)//60:02d}:{int(ts)%60:02d}]", "type": "text"},
                    {"text": None, "type": "image"},
                    {"text": "\n", "type": "text"},
                ]
            )
    return contents


@register_model("aria")
class Aria(BasicModel):
    def __init__(
        self,
        model_path: str = "rhymes-ai/Aria",
        num_segments: int = 128
    ):
        super().__init__(model_path)
        self.num_frames = 128
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self._processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
    def set_frame_num(self, new_num):
        self.num_frames = new_num

    def generate_until(self, visual, text) -> str:

        video_path = visual
        frames, frame_timestamps = load_video(video_path, num_frames=self.num_frames)
        contents = get_placeholders_for_videos(frames, frame_timestamps)

        messages = [
            {
                "role": "user",
                "content": [
                    *contents,
                    {"text": text, "type": "text"},
                ],
            }
        ]

        text = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(
            text=text, images=frames, return_tensors="pt", max_image_size=490
        )
        inputs["pixel_values"] = inputs["pixel_values"].to(self._model.dtype)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            output = self._model.generate(
                **inputs,
                max_new_tokens=2048,
                stop_strings=["<|im_end|>"],
                tokenizer=self._processor.tokenizer,
                do_sample=False,
            )
            output_ids = output[0][inputs["input_ids"].shape[1] :]
            result = self._processor.decode(output_ids, skip_special_tokens=True)

        return result.replace("<|im_end|>", "")

    def generate_until1(self, visual1, visual2, text) -> str:
        video_path = visual1
        frames, frame_timestamps = load_video(video_path, num_frames=self.num_frames)
        contents = get_placeholders_for_videos(frames, frame_timestamps)
        contents.extend(
            [
                {"text": f"Image 1: ", "type": "text"},
                {"text": None, "type": "image"},
                {"text": "\n", "type": "text"},
            ]
        )
        frames.append(Image.open(visual2).convert("RGB"))
        messages = [
            {
                "role": "user",
                "content": [
                    *contents,
                    {"text": text, "type": "text"},
                ],
            }
        ]

        text = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(
            text=text, images=frames, return_tensors="pt", max_image_size=490
        )
        inputs["pixel_values"] = inputs["pixel_values"].to(self._model.dtype)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            output = self._model.generate(
                **inputs,
                max_new_tokens=2048,
                stop_strings=["<|im_end|>"],
                tokenizer=self._processor.tokenizer,
                do_sample=False,
            )
            output_ids = output[0][inputs["input_ids"].shape[1] :]
            result = self._processor.decode(output_ids, skip_special_tokens=True)

        return result.replace("<|im_end|>", "")

    def generate_until2(self, visual1, visual2, text, target_resolution=None, keep_aspect_ratio = True, min_pixels = None, max_pixels = None) -> str:
        video_path = visual1
        frames, frame_timestamps = load_video(video_path, num_frames=self.num_frames)
        # 调整分辨率
        if target_resolution is not None:
            resized_frames = []
            for frame in frames:
                img = smart_resize_with_target(frame, target_resolution, keep_aspect_ratio, min_pixels, max_pixels)
                resized_frames.append(img)  # 将 PIL 图像转换回 NumPy 数组
            frames = resized_frames
        
        contents = get_placeholders_for_videos(frames, frame_timestamps)
        contents.extend(
            [
                {"text": f"Image 1: ", "type": "text"},
                {"text": None, "type": "image"},
                {"text": "\n", "type": "text"},
            ]
        )

        frames.append(Image.open(visual2).convert("RGB"))

        messages = [
            {
                "role": "user",
                "content": [
                    *contents,
                    {"text": text, "type": "text"},
                ],
            }
        ]

        text = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(
            text=text, images=frames, return_tensors="pt", max_image_size=490
        )
        inputs["pixel_values"] = inputs["pixel_values"].to(self._model.dtype)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            output = self._model.generate(
                **inputs,
                max_new_tokens=2048,
                stop_strings=["<|im_end|>"],
                tokenizer=self._processor.tokenizer,
                do_sample=False,
            )
            output_ids = output[0][inputs["input_ids"].shape[1] :]
            result = self._processor.decode(output_ids, skip_special_tokens=True)

        return result.replace("<|im_end|>", "")

    def generate_video_only(self, visual, text, num_frames) -> str:
        video_path = visual
        frames, frame_timestamps = load_video(video_path, num_frames=self.num_frames)
        contents = get_placeholders_for_videos(frames, frame_timestamps)

        messages = [
            {
                "role": "user",
                "content": [
                    *contents,
                    {"text": text, "type": "text"},
                ],
            }
        ]

        text = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(
            text=text, images=frames, return_tensors="pt", max_image_size=490
        )
        inputs["pixel_values"] = inputs["pixel_values"].to(self._model.dtype)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            output = self._model.generate(
                **inputs,
                max_new_tokens=2048,
                stop_strings=["<|im_end|>"],
                tokenizer=self._processor.tokenizer,
                do_sample=False,
            )
            output_ids = output[0][inputs["input_ids"].shape[1]:]
            result = self._processor.decode(output_ids, skip_special_tokens=True)

        return result.replace("<|im_end|>", "")
    
    
    def generate_video_only_res(self, visual, text, target_resolution=None,
                            keep_aspect_ratio=True, min_pixels=None, max_pixels=None) -> str:
        video_path = visual
        frames, frame_timestamps = load_video(video_path, num_frames=self.num_frames)

        if target_resolution is not None:
            resized_frames = []
            for frame in frames:
                img = smart_resize_with_target(frame, target_resolution, keep_aspect_ratio, min_pixels, max_pixels)
                resized_frames.append(img)
            frames = resized_frames

        contents = get_placeholders_for_videos(frames, frame_timestamps)
        messages = [
            {
                "role": "user",
                "content": [
                    *contents,
                    {"text": text, "type": "text"},
                ],
            }
        ]

        text = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(
            text=text, images=frames, return_tensors="pt", max_image_size=490
        )
        inputs["pixel_values"] = inputs["pixel_values"].to(self._model.dtype)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            output = self._model.generate(
                **inputs,
                max_new_tokens=2048,
                stop_strings=["<|im_end|>"],
                tokenizer=self._processor.tokenizer,
                do_sample=False,
            )
            output_ids = output[0][inputs["input_ids"].shape[1]:]
            result = self._processor.decode(output_ids, skip_special_tokens=True)

        return result.replace("<|im_end|>", "")
