from abc import ABC, abstractmethod
from typing import Dict, Any, Union
import torch


class VLMBaseModule(ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def get_vlm_key(self):
        pass

    @abstractmethod
    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        pass

    def post_model_init(self, model, processing_class):
        pass

    def is_embeds_input(self):
        return False
    
    @abstractmethod
    def get_processing_class(self):
        pass

    @abstractmethod
    def get_vision_modules_keywords(self):
        pass

    @abstractmethod
    def get_custom_multimodal_keywords(self):
        pass

    @abstractmethod
    def get_non_generate_params(self):
        pass

    @abstractmethod
    def get_custom_processing_keywords(self):
        pass

    @abstractmethod
    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        pass
    
    @abstractmethod
    def prepare_model_inputs(self, processing_class, prompts_text, images, return_tensors, padding, padding_side, add_special_tokens):
        pass