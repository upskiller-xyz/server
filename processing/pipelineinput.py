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
