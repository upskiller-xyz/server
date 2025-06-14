from __future__ import annotations
from dataclasses import dataclass
import json
import logging

from ..pipelineinput import PipelineInput
from ..pipeline_config import PipelineConfig


class Step:
    name = "generic_step"

    @classmethod
    def run(cls,  inp:PipelineInput, config:PipelineConfig=PipelineConfig.default())->PipelineInput:
        return cls._run(inp, config)
    
    @classmethod
    def _run(cls, inp:PipelineInput, config:PipelineConfig=PipelineConfig.default())->PipelineInput:
        
        return inp
    
class PredictStep(Step):
    name = "predict_step"

    @classmethod
    def _run(cls, inp:PipelineInput, config:PipelineConfig=PipelineConfig.default())->PipelineInput:
        
        return inp
    

class ImageCombineStep(Step):
    name = "imagecombine_step"

    @classmethod
    def _run(cls, inp:PipelineInput, config:PipelineConfig=PipelineConfig.default())->PipelineInput:
        
        return inp
    
class ImageAlignStep(Step):
    name = "imagealign_step"

    @classmethod
    def _run(cls, inp:PipelineInput, config:PipelineConfig=PipelineConfig.default())->PipelineInput:
        
        return inp