from __future__ import annotations
from dataclasses import dataclass, field
import logging
import tensorflow as tf
import numpy as np

@dataclass
class Matrix:
    rotation: float
    translation: float

@dataclass
class PredictionInput:
    id: str
    image: tf.Tensor = field(repr=False)
    params: Matrix

    def __post_init__(self):
        if isinstance(self.image, np.ndarray):
            self.image = tf.convert_to_tensor(self.image)
        elif isinstance(self.image, (bytes, bytearray)):
            self.image = tf.io.decode_image(self.image)

