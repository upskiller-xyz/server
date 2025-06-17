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
class GetDfPipelineInput(PipelineInput):
    value: tf.Tensor

    @classmethod
    def build(cls, bytestring:str)->GetDfPipelineInput:
        img = ImageManager.get_image(bytestring)
        return cls(img)