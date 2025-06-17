from __future__ import annotations
from dataclasses import dataclass
import json
import logging
import tensorflow as tf

from .image_manager import ImageManager

logger = logging.getLogger(__name__)    

@dataclass
class PipelineInput:
    value: int

    @classmethod
    def default(cls)->PipelineInput:
        return PipelineInput(0)
    
    @classmethod
    def build(cls, value:int)->PipelineInput:
        return cls(value)
    
@dataclass
class GetOneDfPipelineInput(PipelineInput):
    value: tf.Tensor

    @classmethod
    def build(cls, bytestring:str)->GetOneDfPipelineInput:
        img = ImageManager.get_image(bytestring)
        return cls(img)
    
@dataclass
class GetDfPipelineInput(PipelineInput):
    value: list[tf.Tensor]

    @classmethod
    def build(cls, bytestrings:list[str])->GetDfPipelineInput:
        return [ImageManager.get_image(b) for b in bytestrings]