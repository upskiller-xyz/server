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
from dataclasses import dataclass
import json
import logging
import tensorflow as tf

from .image_manager import ImageManager
from .prediction_input import PredictionInput, Matrix

logger = logging.getLogger(__name__)


@dataclass
class PipelineInput:
    value: int

    @classmethod
    def default(cls) -> PipelineInput:
        return PipelineInput(0)

    @classmethod
    def build(cls, value: int) -> PipelineInput:
        return cls(value)


@dataclass
class GetOneDfPipelineInput(PipelineInput):
    value: PredictionInput

    @classmethod
    def build(
        cls, bytestring: str, id: str = "", params: Matrix = Matrix(0, (0, 0, 0))
    ) -> GetOneDfPipelineInput:
        img = ImageManager.get_image(bytestring)
        return cls(PredictionInput(id, img, params))


@dataclass
class GetDfPipelineInput(PipelineInput):
    value: list[PredictionInput]
