from __future__ import annotations
from dataclasses import dataclass
import json
import logging

from .steps import step as st
from . import pipelineinput as inpt
from .processor import MultiThreader

logger = logging.getLogger(__name__)


class Pipeline(st.Step):
    steps: list[st.Step] = []

    @classmethod
    def run(cls, inp: inpt.PipelineInput) -> inpt.PipelineInput:
        return cls._run(inp)

    @classmethod
    def _run(cls, inp: inpt.PipelineInput) -> inpt.PipelineInput:
        for step in cls.steps:
            try:
                inp = step.run(inp)
            except Exception as e:
                logger.exception("Step {}; error {}".format(step.__name__, e))
        return inp


class GetOneDfPipeline(Pipeline):
    steps = [st.ImagePreprocessStep, st.PredictStep, st.ImageAlignStep]


class GetDfPipeline(Pipeline):
    steps = [
        GetOneDfPipeline,
        st.ImageCombineStep,
        st.ImageScaleStep,
    ]

    @classmethod
    def _run(cls, inp: inpt.PipelineInput) -> inpt.PipelineInput:
        inp = inpt.GetDfPipelineInput(
            [x for x in MultiThreader.run(cls.steps[0].run, inp.value)]
        )
        for step in cls.steps[1:]:
            try:
                inp = step.run(inp)
            except Exception as e:
                logger.exception("Step {}; error {}".format(step.__name__, e))
        return inp


if __name__ == "__main__":
    pass
