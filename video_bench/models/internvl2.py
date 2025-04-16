import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from video_bench.models.basic_model import BasicModel
from video_bench.registry import register_model
from video_bench.res_smart import smart_resize_with_target

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=False, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values, pixel_values.shape[0]

# video multi-round conversation (视频多轮对话)
def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list, pixel_values_list

def load_video_res(video_path, bound=None, input_size=448, max_num=1, num_segments=32, target_resolution=None, keep_aspect_ratio = True, min_pixels = None, max_pixels = None):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')

        if target_resolution is not None:
            img = smart_resize_with_target(img, target_resolution, keep_aspect_ratio, min_pixels, max_pixels)
        
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list, pixel_values_list

@register_model("internvl2")
class Internvl2(BasicModel):
    def __init__(
        self, model_path: str="OpenGVLab/InternVL2-8B",
        max_num: str = 1,
    ):
        self.num_segments = 128
        self.max_num = max_num
        super().__init__(model_path)
        self._model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            # use_flash_attn=True,
            trust_remote_code=True).eval().cuda()
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    def set_frame_num(self, new_num):
        self.num_segments = new_num

    def generate_until(self, visual, text) -> str:
        generation_config = dict(max_new_tokens=1024, do_sample=False)
        video_path = visual
        pixel_values, num_patches_list, pixel_values_list = load_video(video_path, num_segments=8, max_num=1)
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
        question = video_prefix + text
        response, history = self._model.chat(self._tokenizer, pixel_values, question, generation_config,
                                    num_patches_list=num_patches_list, history=None, return_history=True)

        return response

    def generate_until1(self, visual1, visual2, text, nframes) -> str:
        generation_config = dict(max_new_tokens=1024, do_sample=False)
        pixel_vi = []
        video_path = visual1
        pixel_values, num_patches_list, pixel_values_list = load_video(video_path, num_segments=self.num_segments, max_num=self.max_num)
        pixel_image, item = load_image(visual2)
        pixel_values_list.append(pixel_image)
        pixel_vi = torch.cat(pixel_values_list)
        pixel_values = pixel_vi.to(torch.bfloat16).cuda()
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
        question = video_prefix + "\nImage1: <image>\n" + text
        num_patches_list.append(item)
        response, history = self._model.chat(self._tokenizer, pixel_values, question, generation_config,
                                    num_patches_list=num_patches_list, history=None, return_history=True)

        return response

    def generate_until2(self, visual1, visual2, text, target_resolution=None, keep_aspect_ratio = True, min_pixels = None, max_pixels = None) -> str:
        generation_config = dict(max_new_tokens=1024, do_sample=False)
        pixel_vi = []
        video_path = visual1
        pixel_values, num_patches_list, pixel_values_list = load_video_res(video_path, num_segments=self.num_segments, max_num=self.max_num, target_resolution=target_resolution, keep_aspect_ratio=keep_aspect_ratio, min_pixels=min_pixels, max_pixels=max_pixels)
        pixel_image, item = load_image(visual2)
        pixel_values_list.append(pixel_image)
        pixel_vi = torch.cat(pixel_values_list)
        pixel_values = pixel_vi.to(torch.bfloat16).cuda()
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
        question = video_prefix + "\nImage1: <image>\n" + text
        num_patches_list.append(item)
        response, history = self._model.chat(self._tokenizer, pixel_values, question, generation_config,
                                    num_patches_list=num_patches_list, history=None, return_history=True)

        return response

    def generate_video_only(self, visual, text, nframe) -> str:
        generation_config = dict(max_new_tokens=1024, do_sample=False)
        video_path = visual
        pixel_values, num_patches_list, _ = load_video(
            video_path,
            num_segments=self.num_segments,
            max_num=self.max_num
        )
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
        question = video_prefix + text
        response, history = self._model.chat(
            self._tokenizer,
            pixel_values,
            question,
            generation_config,
            num_patches_list=num_patches_list,
            history=None,
            return_history=True,
        )
        return response

    def generate_video_only_res(self, visual, text, target_resolution=None, keep_aspect_ratio=True, min_pixels=None, max_pixels=None) -> str:
        generation_config = dict(max_new_tokens=1024, do_sample=False)
        video_path = visual
        pixel_values, num_patches_list, _ = load_video_res(
            video_path,
            num_segments=self.num_segments,
            max_num=self.max_num,
            target_resolution=target_resolution,
            keep_aspect_ratio=keep_aspect_ratio,
            min_pixels=min_pixels,
            max_pixels=max_pixels
        )
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
        question = video_prefix + text
        response, history = self._model.chat(
            self._tokenizer,
            pixel_values,
            question,
            generation_config,
            num_patches_list=num_patches_list,
            history=None,
            return_history=True,
        )
        return response