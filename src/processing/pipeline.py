# Daylight Factor Estimation Server
# Copyright (C) 2024 BIMTech Innovations AB (developed by the Upskiller group)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU GPL v3.0 along with this program.
# If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations
import cv2
from dataclasses import dataclass
from datetime import datetime
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
    @classmethod
    def _run(cls, inp: inpt.PipelineInput) -> inpt.PipelineInput:
        for step in cls.steps:
            try:
                # _ = cv2.imwrite("../assets/pre_{}_{}.png".format(step.__name__, inp.value.id), inp.value.np_image * 255)
                inp = step.run(inp)
                # _ = cv2.imwrite("../assets/{}_{}.png".format(step.__name__, inp.value.id), inp.value.np_image)
            except Exception as e:
                logger.exception("Step {}; error {}".format(step.__name__, e))
        return inp


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
