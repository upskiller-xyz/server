from __future__ import annotations
from dataclasses import dataclass

from ..pipelineinput import PipelineInput
from ..pipeline_config import PipelineConfig
from ..inference import Inference
from ..image_transformer import ImageTransformer

class Step:

    @classmethod
    def run(cls,  inp:PipelineInput, config:PipelineConfig=PipelineConfig.default())->PipelineInput:
        return cls._run(inp, config)
    
    @classmethod
    def _run(cls, inp:PipelineInput, config:PipelineConfig=PipelineConfig.default())->PipelineInput:
        
        return inp
    
class PredictStep(Step):

    @classmethod
    def _run(cls, inp:PipelineInput, config:PipelineConfig=PipelineConfig.default())->PipelineInput:
        return Inference.run(inp.value)
    

class ImageCombineStep(Step):

    @classmethod
    def _run(cls, inp:PipelineInput, config:PipelineConfig=PipelineConfig.default())->PipelineInput:
        
        return inp
    
class ImageAlignStep(Step):

    @classmethod
    def _run(cls, inp:PipelineInput, config:PipelineConfig=PipelineConfig.default())->PipelineInput:
        return ImageTransformer.run(inp)