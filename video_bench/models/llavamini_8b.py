import argparse
import torch

from llavamini.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llavamini.conversation import conv_templates, SeparatorStyle
from llavamini.model.builder import load_pretrained_model
from llavamini.utils import disable_torch_init
from llavamini.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image
import numpy as np
from decord import VideoReader, cpu
import requests
from PIL import Image
from io import BytesIO
import re
import torch
import time
import re 

from video_bench.models.basic_model import BasicModel
from video_bench.registry import register_model
from video_bench.res_smart import smart_resize_with_target

def image_parser(image_file):
    out = image_file.split(",")
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out



def load_video(vis_path, n_clips=1, num_frm=None):
    """
    Load video frames from a video file.

    Parameters:
    vis_path (str): Path to the video file.
    n_clips (int): Number of clips to extract from the video. Defaults to 1.
    num_frm (int): Number of frames to extract from each clip. Defaults to 100.

    Returns:
    list: List of PIL.Image.Image objects representing video frames.
    """

    # Load video with VideoReader
    vr = VideoReader(vis_path, ctx=cpu(0))
    total_frame_num = len(vr)

    # Currently, this function supports only 1 clip
    assert n_clips == 1

    if num_frm==None:
        fps=vr.get_avg_fps()
        num_frm=int(total_frame_num//fps)

    # Calculate total number of frames to extract
    total_num_frm = min(total_frame_num, num_frm)
    # Get indices of frames to extract
    frame_idx = get_seq_frames(total_frame_num, total_num_frm)
    # Extract frames as numpy array
    img_array = vr.get_batch(frame_idx).asnumpy()
    # Set target image height and width
    target_h, target_w = 336, 336   
    # If image shape is not as target, resize it
    if img_array.shape[-3] != target_h or img_array.shape[-2] != target_w:
        img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
        img_array = torch.nn.functional.interpolate(img_array, size=(target_h, target_w))
        img_array = img_array.permute(0, 2, 3, 1).to(torch.uint8).numpy()

    # Reshape array to match number of clips and frames
    img_array = img_array.reshape(
        (n_clips, total_num_frm, img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))
    # Convert numpy arrays to PIL Image objects
    clip_imgs = [Image.fromarray(img_array[0, j]) for j in range(total_num_frm)]

    return clip_imgs


def load_video_res(vis_path, n_clips=1, num_frm=None, target_resolution=None, keep_aspect_ratio = True, min_pixels = None, max_pixels = None):
    """
    Load video frames from a video file.

    Parameters:
    vis_path (str): Path to the video file.
    n_clips (int): Number of clips to extract from the video. Defaults to 1.
    num_frm (int): Number of frames to extract from each clip. Defaults to 100.

    Returns:
    list: List of PIL.Image.Image objects representing video frames.
    """

    # Load video with VideoReader
    vr = VideoReader(vis_path, ctx=cpu(0))
    total_frame_num = len(vr)

    # Currently, this function supports only 1 clip
    assert n_clips == 1

    if num_frm==None:
        fps=vr.get_avg_fps()
        num_frm=int(total_frame_num//fps)

    # Calculate total number of frames to extract
    total_num_frm = min(total_frame_num, num_frm)
    # Get indices of frames to extract
    frame_idx = get_seq_frames(total_frame_num, total_num_frm)
    # Extract frames as numpy array
    img_array = vr.get_batch(frame_idx).asnumpy()
    # Set target image height and width
    target_h, target_w = 336, 336   
    # If image shape is not as target, resize it
    if img_array.shape[-3] != target_h or img_array.shape[-2] != target_w:
        img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
        img_array = torch.nn.functional.interpolate(img_array, size=(target_h, target_w))
        img_array = img_array.permute(0, 2, 3, 1).to(torch.uint8).numpy()

    # Reshape array to match number of clips and frames
    img_array = img_array.reshape(
        (n_clips, total_num_frm, img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))
    # Convert numpy arrays to PIL Image objects
    if target_resolution is not None:
        if max_pixels is not None and min_pixels is not None:
            clip_imgs = [smart_resize_with_target(Image.fromarray(img_array[0, j]), target_resolution, keep_aspect_ratio, min_pixels, max_pixels) for j in range(total_num_frm)]
        else:
            clip_imgs = [smart_resize_with_target(Image.fromarray(img_array[0, j]), target_resolution, keep_aspect_ratio) for j in range(total_num_frm)]
    else:
        clip_imgs = [Image.fromarray(img_array[0, j]) for j in range(total_num_frm)]

    return clip_imgs

def get_seq_frames(total_num_frames, desired_num_frames):
    """
    Calculate the indices of frames to extract from a video.

    Parameters:
    total_num_frames (int): Total number of frames in the video.
    desired_num_frames (int): Desired number of frames to extract.

    Returns:
    list: List of indices of frames to extract.
    """

    # Calculate the size of each segment from which a frame will be extracted
    seg_size = float(total_num_frames - 1) / desired_num_frames

    seq = []
    for i in range(desired_num_frames):
        # Calculate the start and end indices of each segment
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))

        # Append the middle index of the segment to the list
        seq.append((start + end) // 2)

    return seq


def split_image(image, n=2):
    if n==1: return [image]
    width, height = image.size
    block_width = width // n
    block_height = height // n

    blocks = []

    for i in range(n):
        for j in range(n):
            left = j * block_width
            upper = i * block_height
            right = (j + 1) * block_width
            lower = (i + 1) * block_height
            block = image.crop((left, upper, right, lower))
            blocks.append(block)
    blocks.append(image)

    return blocks


@register_model("llavamini")
class LLaVAMini(BasicModel):
    def __init__(
        self, model_path: str="ICTNLP/llava-mini-llama-3.1-8b",
    ):
        super().__init__(model_path)
        self.model_name = "llava-mini"
        self._tokenizer, self._model, self._image_processor, self._context_len = load_pretrained_model(
            model_path, None, "llava-mini",load_8bit="store_true"
        )
        self.max_num_frames = 128
        self.conv_mode = "llava_llama_3_1"
    
    def set_frame_num(self, new_num):
        self.max_num_frames = new_num

    def _get_conv_mode(self):
        if "llama-2" in self.model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in self.model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in self.model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in self.model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        if self.conv_mode is not None and conv_mode != self.conv_mode:
            print(
                "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                    conv_mode, self.conv_mode, self.conv_mode
                )
            )
        else:
            self.conv_mode = conv_mode
        return conv_mode

    def generate_until(self, visual, text) -> str:
        # Model
        disable_torch_init()

        qs = text
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if self._model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self._model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        if "llama-2" in self.model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in self.model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in self.model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in self.model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        if self.conv_mode is not None and conv_mode != self.conv_mode:
            print(
                "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                    conv_mode, self.conv_mode, self.conv_mode
                )
            )
        else:
            self.conv_mode = conv_mode

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        video_file = visual

        assert video_file is not None

        if video_file is not None:
            video_path = video_file
            video_frames = load_video(video_path, num_frm=self.max_num_frames)
            temporal_len = len(video_frames)
            N = getattr(self._model.config, 'resolution_ratio', 1)
            images = []
            for video_frame in video_frames:
                images.extend(split_image(video_frame, n=N))

            image_tensor = self._image_processor.preprocess(images, return_tensors='pt')['pixel_values']
            image_tensor = image_tensor.to(self._model.device, dtype=torch.float16).unsqueeze(0)

            bsz, N2_x_temporal, rgb, height, width = image_tensor.size()
            images_tensor = image_tensor.view(bsz, temporal_len, -1, rgb, height, width)
            print(image_tensor.size())


        input_ids = (
            tokenizer_image_token(prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        with torch.inference_mode():
            output_ids = self._model.generate(
                input_ids,
                images=images_tensor,
                do_sample=True,
                num_beams=1,
                max_new_tokens=512,
                use_cache=True,
            )
        outputs = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs


    def generate_until1(self, visual1, visual2, text) -> str:
        # Model
        disable_torch_init()

        qs = text
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if self._model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self._model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        if "llama-2" in self.model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in self.model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in self.model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in self.model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        if self.conv_mode is not None and conv_mode != self.conv_mode:
            print(
                "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                    conv_mode, self.conv_mode, self.conv_mode
                )
            )
        else:
            self.conv_mode = conv_mode

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        video_file = visual1
        image_file = visual2

        assert video_file is not None or image_file is not None

        if video_file is not None:
            video_path = video_file
            video_frames = load_video(video_path, num_frm=self.max_num_frames)
            image_files = image_parser(image_file)
            images = load_images(image_files)
            video_frames.append(images[0])
            temporal_len = len(video_frames)
            N = getattr(self._model.config, 'resolution_ratio', 1)
            images = []
            for video_frame in video_frames:
                images.extend(split_image(video_frame, n=N))

            image_tensor = self._image_processor.preprocess(images, return_tensors='pt')['pixel_values']
            image_tensor = image_tensor.to(self._model.device, dtype=torch.float16).unsqueeze(0)

            bsz, N2_x_temporal, rgb, height, width = image_tensor.size()
            images_tensor = image_tensor.view(bsz, temporal_len, -1, rgb, height, width)
            print(image_tensor.size())


        input_ids = (
            tokenizer_image_token(prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        with torch.inference_mode():
            output_ids = self._model.generate(
                input_ids,
                images=images_tensor,
                do_sample=True,
                num_beams=1,
                max_new_tokens=512,
                use_cache=True,
            )
        outputs = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs


    def generate_until2(self, visual1, visual2, text, target_resolution=None, keep_aspect_ratio = True, min_pixels = None, max_pixels = None) -> str:
        # Model
        disable_torch_init()

        qs = text
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if self._model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self._model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        if "llama-2" in self.model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in self.model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in self.model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in self.model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        if self.conv_mode is not None and conv_mode != self.conv_mode:
            print(
                "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                    conv_mode, self.conv_mode, self.conv_mode
                )
            )
        else:
            self.conv_mode = conv_mode

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        video_file = visual1
        image_file = visual2

        assert video_file is not None or image_file is not None

        if video_file is not None:
            video_path = video_file
            video_frames = load_video_res(video_path, self.max_num_frames, target_resolution, keep_aspect_ratio, min_pixels, max_pixels)
            image_files = image_parser(image_file)
            images = load_images(image_files)
            video_frames.append(images[0])
            temporal_len = len(video_frames)
            N = getattr(self._model.config, 'resolution_ratio', 1)
            images = []
            for video_frame in video_frames:
                images.extend(split_image(video_frame, n=N))

            image_tensor = self._image_processor.preprocess(images, return_tensors='pt')['pixel_values']
            image_tensor = image_tensor.to(self._model.device, dtype=torch.float16).unsqueeze(0)

            bsz, N2_x_temporal, rgb, height, width = image_tensor.size()
            images_tensor = image_tensor.view(bsz, temporal_len, -1, rgb, height, width)
            print(image_tensor.size())


        input_ids = (
            tokenizer_image_token(prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        with torch.inference_mode():
            output_ids = self._model.generate(
                input_ids,
                images=images_tensor,
                do_sample=True,
                num_beams=1,
                max_new_tokens=512,
                use_cache=True,
            )
        outputs = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs
    
    def generate_video_only(self, visual1: str, text: str,nframes) -> str:
        disable_torch_init()

        qs = text
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if self._model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self._model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv_mode = self._get_conv_mode()
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        video_path = visual1
        video_frames = load_video(video_path, num_frm=self.max_num_frames)
        temporal_len = len(video_frames)
        N = getattr(self._model.config, 'resolution_ratio', 1)

        images = []
        for video_frame in video_frames:
            images.extend(split_image(video_frame, n=N))

        image_tensor = self._image_processor.preprocess(images, return_tensors='pt')['pixel_values']
        image_tensor = image_tensor.to(self._model.device, dtype=torch.float16).unsqueeze(0)
        bsz, N2_x_temporal, rgb, height, width = image_tensor.size()
        images_tensor = image_tensor.view(bsz, temporal_len, -1, rgb, height, width)

        input_ids = tokenizer_image_token(prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

        with torch.inference_mode():
            output_ids = self._model.generate(
                input_ids,
                images=images_tensor,
                do_sample=True,
                num_beams=1,
                max_new_tokens=512,
                use_cache=True,
            )

        outputs = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs
    def generate_video_only_res(self, visual1: str, text: str, target_resolution=None, keep_aspect_ratio=True, min_pixels=None, max_pixels=None) -> str:
        disable_torch_init()

        qs = text
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if self._model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self._model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv_mode = self._infer_conv_mode()
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        video_path = visual1
        video_frames = load_video_res(video_path, self.max_num_frames, target_resolution, keep_aspect_ratio, min_pixels, max_pixels)
        temporal_len = len(video_frames)
        N = getattr(self._model.config, 'resolution_ratio', 1)

        images = []
        for video_frame in video_frames:
            images.extend(split_image(video_frame, n=N))

        image_tensor = self._image_processor.preprocess(images, return_tensors='pt')['pixel_values']
        image_tensor = image_tensor.to(self._model.device, dtype=torch.float16).unsqueeze(0)
        bsz, N2_x_temporal, rgb, height, width = image_tensor.size()
        images_tensor = image_tensor.view(bsz, temporal_len, -1, rgb, height, width)

        input_ids = tokenizer_image_token(prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

        with torch.inference_mode():
            output_ids = self._model.generate(
                input_ids,
                images=images_tensor,
                do_sample=True,
                num_beams=1,
                max_new_tokens=512,
                use_cache=True,
            )

        outputs = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs
