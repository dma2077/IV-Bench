import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoModel, AutoImageProcessor
import torch
from video_bench.models.basic_model import BasicModel
from video_bench.registry import register_model


@register_model("videollama3")
class VideoLlama3(BasicModel):
    def __init__(
        self,
        model_path: str = "DAMO-NLP-SG/VideoLLaMA3-7B",
    ):
        super().__init__(model_path)

        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).eval()
        self._processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self._config = self._model.config
        self.max_num_frames = 128

    def set_frame_num(self, new_num):
        self.max_num_frames = new_num
        print(f"set max frames:{self.max_num_frames}!!!")

    def generate_until1(self, visual1, visual2, text) -> str:
        # Video conversation
        video_path = visual1
        image_path = visual2
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": {"video_path": video_path, "fps": 1, "max_frames": self.max_num_frames}},
                    {"type": "image", "image": {"image_path": image_path}},
                    {"type": "text", "text": text},
                ]
            },
        ]
        inputs = self._processor(conversation=conversation, return_tensors="pt")
        inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        output_ids = self._model.generate(**inputs, max_new_tokens=128)
        response = self._processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return response


    def generate_video_only(self, visual1, text, num_frames) -> str:
        # Video conversation
        video_path = visual1
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": {"video_path": video_path, "fps": 1, "max_frames": self.max_num_frames}},
                    {"type": "text", "text": text},
                ]
            },
        ]
        inputs = self._processor(conversation=conversation, return_tensors="pt")
        inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        output_ids = self._model.generate(**inputs, max_new_tokens=128)
        response = self._processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return response
