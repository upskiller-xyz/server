from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import os
import tensorflow as tf
import logging

from .prediction_input import PredictionInput
from .utils import TARGET_SIZE

logger = logging.Logger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ML_MODEL_FILENAME = 'generator.keras' # pix2pix modell
ML_MODEL_GCS_URI = os.getenv('ML_MODEL_GCS_URI', 'gs://df_experiments/fa3a571f-b1da-47db-8b0e-db7caaec9e8e/best_generator.keras')

class Inference:
    model = None

    @classmethod
    def init(cls, model_path:str=ML_MODEL_GCS_URI)->bool:
        try:
            cls.model = tf.keras.models.load_model(model_path, compile=False)
            return True
        except Exception as e:
            logger.exception(e)
            return False

    @classmethod
    def run(cls, inp:PredictionInput)->PredictionInput:
        if not cls.model:
            _init = cls.init()
            if not _init:
                return PredictionInput(inp.id, tf.zeros(tf.shape(inp.image)), inp.params)
        img = tf.reshape(inp.image, (-1, TARGET_SIZE, TARGET_SIZE, 3))
        res = cls.model(img, training=False)
        return PredictionInput(inp.id, res, inp.params)
