from __future__ import annotations
from dataclasses import dataclass
import json
import logging

from steps.step import Step
from .pipelineinput import PipelineInput
from .pipeline_config import PipelineConfig

logger = logging.getLogger(__name__)    


class Pipeline:
    steps:list[Step] = []

    @classmethod
    def run(cls, inp:PipelineInput, PipelineConfig:PipelineConfig=PipelineConfig.default())->PipelineInput:
        return cls._run(PipelineConfig

, inp)
    
    @classmethod
    def _run(cls, inp:PipelineInput, PipelineConfig:PipelineConfig=PipelineConfig.default())->PipelineInput:
        for step in cls.steps:
            try:
                inp = step.run(inp, PipelineConfig)
            except Exception as e:
                logger.exception("Step {}; error {}".format(step.name, e))
        return inp

if __name__ == '__main__':
    pass
