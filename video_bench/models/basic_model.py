import torch


class BasicModel():
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate_until(self, inputs: torch.Tensor) -> str:
        pass
