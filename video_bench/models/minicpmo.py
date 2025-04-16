import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from decord import VideoReader, cpu  # pip install decord

from video_bench.models.basic_model import BasicModel
from video_bench.registry import register_model
from video_bench.res_smart import smart_resize_with_target

MAX_NUM_FRAMES = 256 


def encode_video(video_path: str, num_segments):

    def uniform_sample(lst, n):

        gap = len(lst) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [lst[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1) 
    frame_idx = list(range(0, len(vr), sample_fps))

    frame_idx = uniform_sample(frame_idx, min(num_segments, MAX_NUM_FRAMES))

    frames_array = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(f.astype("uint8")) for f in frames_array]
    return frames


@register_model("minicpmo")
class MiniCPMO(BasicModel):
    def __init__(
        self, model_path: str = "openbmb/MiniCPM-o-2_6", attn_impl: str = "sdpa",
    ):

        super().__init__(model_path)
        self.num_segments = 128
        self._model = (
            AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                # attn_implementation=attn_impl,
                torch_dtype=torch.bfloat16,
            )
            .eval()
            .cuda()
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
    
    def set_frame_num(self, new_num):
        self.num_segments = new_num

    def generate_until(self, visual: str, text: str) -> str:
        frames = encode_video(visual, self.num_segments)
        msgs = [
            {"role": "user", "content": frames + [text]},
        ]
        params = {
            "use_image_id": False,
            "max_slice_nums": 2,  
            "do_sample": False,
            "max_new_token": 1024,
        }
        answer = self._model.chat(
            image=None, msgs=msgs, tokenizer=self._tokenizer, **params
        )
        return answer

    def generate_until1(self, visual1: str, visual2: str, text: str) -> str:

        frames = encode_video(visual1, self.num_segments)
        image = [Image.open(visual2).convert("RGB")]

        all_images = frames + image
        msgs = [
            {"role": "user", "content": all_images + [text]},
        ]
        params = {
            "use_image_id": False,
            "max_slice_nums": 1,  
            "do_sample": False,
            "max_new_token": 1024,
        }
        answer = self._model.chat(
            image=None, msgs=msgs, tokenizer=self._tokenizer, **params
        )
        return answer


    def generate_until3(self, visual1: str, visual2: str, text: str) -> str:

        frames = encode_video(visual1, self.num_segments)
        image = [Image.open(visual2).convert("RGB")]

        all_images = image + frames
        msgs = [
            {"role": "user", "content": all_images + [text]},
        ]
        params = {
            "use_image_id": False,
            "max_slice_nums": 1,  
            "do_sample": False,
            "max_new_token": 1024,
        }
        answer = self._model.chat(
            image=None, msgs=msgs, tokenizer=self._tokenizer, **params
        )
        return answer

    def generate_until2(self, visual1: str, visual2: str, text: str, target_resolution=None, keep_aspect_ratio = True, min_pixels = None, max_pixels = None) -> str:

        frames = encode_video(visual1, self.num_segments)

        if target_resolution is not None:
            resized_frames = []
            for frame in frames:
                img = smart_resize_with_target(frame, target_resolution, keep_aspect_ratio, min_pixels, max_pixels)
                resized_frames.append(img) 
            frames = resized_frames

        image = [Image.open(visual2).convert("RGB")]

        all_images = frames + image
        msgs = [
            {"role": "user", "content": all_images + [text]},
        ]
        params = {
            "use_image_id": False,
            "max_slice_nums": 1, 
            "do_sample": False,
            "max_new_token": 1024,
        }
        answer = self._model.chat(
            image=None, msgs=msgs, tokenizer=self._tokenizer, **params
        )
        return answer
    
