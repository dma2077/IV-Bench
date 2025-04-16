from longva.model.builder import load_pretrained_model
from longva.mm_utils import tokenizer_image_token, process_images
from longva.constants import IMAGE_TOKEN_INDEX
from PIL import Image
from decord import VideoReader, cpu
import torch
import numpy as np

from video_bench.models.basic_model import BasicModel
from video_bench.registry import register_model
from video_bench.res_smart import smart_resize_with_target

torch.manual_seed(0)

@register_model("longva")
class LongVA(BasicModel):
    def __init__(
        self, model_path: str="lmms-lab/LongVA-7B",
    ):
        super().__init__(model_path)
        self._tokenizer, self._model, self._image_processor, _ = load_pretrained_model(model_path, None, "llava_qwen", device_map="cuda:0")
        self.max_frames_num = 128
    
    def set_frame_num(self, new_num):
        self.max_frames_num = new_num
    
    def generate_until(self, visual, text) -> str:
        gen_kwargs = {"do_sample": True, "temperature": 0.5, "top_p": None, "num_beams": 1, "use_cache": True, "max_new_tokens": 1024}
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n{text}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer_image_token(prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self._model.device)
        vr = VideoReader(visual, ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, self.max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frames = vr.get_batch(frame_idx).asnumpy()
        video_tensor = self._image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(self._model.device, dtype=torch.float16)
        with torch.inference_mode():
            output_ids = self._model.generate(input_ids, images=[video_tensor],  modalities=["video"], **gen_kwargs)
        outputs = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        return outputs

    def generate_until1(self, visual1, visual2, text) -> str:
        gen_kwargs = {"do_sample": True, "temperature": 0.5, "top_p": None, "num_beams": 1, "use_cache": True, "max_new_tokens": 1024}
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n{text}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer_image_token(prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self._model.device)
        vr = VideoReader(visual1, ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, self.max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frames = vr.get_batch(frame_idx).asnumpy()
        image = Image.open(visual2).convert("RGB")

        image_resized = image.resize((frames.shape[2], frames.shape[1]))
        image_np = np.array(image_resized)
        frames_list = list(frames)
        frames_list.append(image_np) 
        frames = np.array(frames_list)

        video_tensor = self._image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(self._model.device, dtype=torch.float16)
        with torch.inference_mode():
            output_ids = self._model.generate(input_ids, images=[video_tensor],  modalities=["video"], **gen_kwargs)
        outputs = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        return outputs
    
    def generate_until2(self, visual1, visual2, text, target_resolution=None, keep_aspect_ratio = True, min_pixels = None, max_pixels = None) -> str:
        gen_kwargs = {"do_sample": True, "temperature": 0.5, "top_p": None, "num_beams": 1, "use_cache": True, "max_new_tokens": 1024}
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n{text}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer_image_token(prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self._model.device)
        vr = VideoReader(visual1, ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, self.max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frames = vr.get_batch(frame_idx).asnumpy()
        
        if target_resolution is not None:
            resized_frames = []
            for frame in frames:
                img = Image.fromarray(frame)  
                img = smart_resize_with_target(img, target_resolution, keep_aspect_ratio, min_pixels, max_pixels)
                resized_frames.append(np.array(img)) 
            frames = np.stack(resized_frames) 
        
        image = Image.open(visual2).convert("RGB")

        image_resized = image.resize((frames.shape[2], frames.shape[1]))
        image_np = np.array(image_resized)
        frames_list = list(frames)
        frames_list.append(image_np) 
        frames = np.array(frames_list)

        video_tensor = self._image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(self._model.device, dtype=torch.float16)
        with torch.inference_mode():
            output_ids = self._model.generate(input_ids, images=[video_tensor],  modalities=["video"], **gen_kwargs)
        outputs = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        return outputs

    def generate_video_only(self, visual1, text,nframes) -> str:
        gen_kwargs = {
            "do_sample": True, "temperature": 0.5, "top_p": None,
            "num_beams": 1, "use_cache": True, "max_new_tokens": 1024
        }

        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" \
                f"<|im_start|>user\n<image>\n{text}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer_image_token(
            prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self._model.device)

        vr = VideoReader(visual1, ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, self.max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frames = vr.get_batch(frame_idx).asnumpy()

        video_tensor = self._image_processor.preprocess(
            frames, return_tensors="pt"
        )["pixel_values"].to(self._model.device, dtype=torch.float16)

        with torch.inference_mode():
            output_ids = self._model.generate(
                input_ids, images=[video_tensor], modalities=["video"], **gen_kwargs
            )
        outputs = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs
    def generate_video_only_res(self, visual1, text, target_resolution=None, keep_aspect_ratio=True, min_pixels=None, max_pixels=None) -> str:
        gen_kwargs = {
            "do_sample": True, "temperature": 0.5, "top_p": None,
            "num_beams": 1, "use_cache": True, "max_new_tokens": 1024
        }

        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" \
                f"<|im_start|>user\n<image>\n{text}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer_image_token(
            prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self._model.device)

        vr = VideoReader(visual1, ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, self.max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frames = vr.get_batch(frame_idx).asnumpy()

        if target_resolution is not None:
            resized_frames = []
            for frame in frames:
                img = Image.fromarray(frame)
                img = smart_resize_with_target(img, target_resolution, keep_aspect_ratio, min_pixels, max_pixels)
                resized_frames.append(np.array(img))
            frames = np.stack(resized_frames)

        video_tensor = self._image_processor.preprocess(
            frames, return_tensors="pt"
        )["pixel_values"].to(self._model.device, dtype=torch.float16)

        with torch.inference_mode():
            output_ids = self._model.generate(
                input_ids, images=[video_tensor], modalities=["video"], **gen_kwargs
            )
        outputs = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs
