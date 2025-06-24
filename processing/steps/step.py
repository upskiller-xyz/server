from __future__ import annotations
from dataclasses import dataclass
import logging
import cv2
import tensorflow as tf

from .. import pipelineinput as inpt
from ..prediction_input import PredictionInput
from ..inference import Inference
from ..image_transformer import ImageTransformer
from ..utils import TARGET_SIZE

logger = logging.getLogger(__name__)


class Step:

    @classmethod
    def run(cls, inp: inpt.PipelineInput) -> inpt.PipelineInput:
        logger.debug("Running step {}".format(cls.__name__))
        return cls._run(inp)

    @classmethod
    def _run(cls, inp: inpt.PipelineInput) -> inpt.PipelineInput:

        return inp


class PredictStep(Step):

    @classmethod
    def _run(cls, inp: inpt.PipelineInput) -> inpt.PipelineInput:
        pred_inp = Inference.run(inp.value)
        return inpt.GetOneDfPipelineInput(pred_inp)


class ImagePreprocessStep(Step):

    @classmethod
    def _run(cls, inp: inpt.PipelineInput) -> inpt.PipelineInput:
        img = ImageTransformer.resize(inp.value.image, TARGET_SIZE, TARGET_SIZE)

        return inpt.GetOneDfPipelineInput(
            PredictionInput(inp.value.id, img, inp.value.params)
        )


class ImageCombineStep(Step):

    @classmethod
    def _run(cls, inp: inpt.GetDfPipelineInput) -> inpt.GetOneDfPipelineInput:

        img = ImageTransformer.combine([i.value.np_image for i in inp.value])

        return inpt.GetOneDfPipelineInput(
            PredictionInput(inp.value[0].value.id, img, inp.value[0].value.params)
        )


class ImageToValuesStep(Step):

    @classmethod
    def _run(cls, inp: inpt.GetDfPipelineInput) -> inpt.GetOneDfPipelineInput:

        img = ImageTransformer.combine([i.value.np_image for i in inp.value])

        return inpt.GetOneDfPipelineInput(
            PredictionInput(inp.value[0].value.id, img, inp.value[0].value.params)
        )


class ImageRotateStep(Step):

    @classmethod
    def _run(cls, inp: inpt.GetOneDfPipelineInput) -> inpt.GetOneDfPipelineInput:
        res = ImageTransformer.run(
            inp.value.np_image, inp.value.params.rotation, inp.value.params.translation
        )
        return inpt.GetOneDfPipelineInput(
            PredictionInput(inp.value.id, res, inp.value.params)
        )


class ImageAlignStep(Step):

    @classmethod
    def _run(cls, inp: inpt.GetOneDfPipelineInput) -> inpt.GetOneDfPipelineInput:

        res = ImageTransformer.run(
            inp.value.np_image, inp.value.params.rotation, inp.value.params.translation
        )
        return inpt.GetOneDfPipelineInput(
            PredictionInput(inp.value.id, res, inp.value.params)
        )


class ImageScaleStep(Step):

    @classmethod
    def _run(cls, inp: inpt.PipelineInput) -> inpt.PipelineInput:
        res = ImageTransformer.resize(inp.value.image)
        return inpt.GetOneDfPipelineInput(
            PredictionInput(inp.value.id, res, inp.value.params)
        )
