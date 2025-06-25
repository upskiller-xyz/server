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
import numpy as np
import os
import tensorflow as tf
import logging

from .image_transformer import ImageTransformer
from .prediction_input import PredictionInput
from .utils import TARGET_SIZE

logger = logging.Logger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ML_MODEL_FILENAME = "generator.keras"  # pix2pix modell
ML_MODEL_GCS_URI = os.getenv(
    "ML_MODEL_GCS_URI",
    "gs://df_experiments/fa3a571f-b1da-47db-8b0e-db7caaec9e8e/best_generator.keras",
)


class Inference:
    model = None

    @classmethod
    def init(cls, model_path: str = ML_MODEL_GCS_URI) -> bool:
        try:
            cls.model = tf.keras.models.load_model(model_path, compile=False)
            return True
        except Exception as e:
            logger.exception(e)
            return False

    @classmethod
    def run(cls, inp: PredictionInput) -> PredictionInput:
        try:
            if not cls.model:
                logger.debug("initializing the model")
                _init = cls.init()
                if not _init:
                    logger.exception(
                        "Model initialization failed. Please check the model path."
                    )
                    return PredictionInput(
                        inp.id,
                        tf.zeros((1, TARGET_SIZE, TARGET_SIZE, 3), dtype=tf.float32),
                        inp.params,
                    )
            img = inp.image
            if np.max(img) > 1.0:
                img = ImageTransformer.norm(img)
            img = tf.reshape(inp.image, (-1, TARGET_SIZE, TARGET_SIZE, 3))
            res = cls.model(img, training=False)
            res = ImageTransformer.denorm(res)
            return PredictionInput(inp.id, res, inp.params)
        except Exception as e:
            logger.exception(e)
            return PredictionInput(
                inp.id,
                tf.zeros((1, TARGET_SIZE, TARGET_SIZE, 3), dtype=tf.float32),
                inp.params,
            )
