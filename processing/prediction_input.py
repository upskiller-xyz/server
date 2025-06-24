from __future__ import annotations
from dataclasses import dataclass, field
import logging
import tensorflow as tf
import numpy as np

from .coord import Coord
from .image_manager import ImageManager


@dataclass
class Matrix:
    rotation: float
    translation: Coord


@dataclass
class PredictionInput:
    id: str
    image: tf.Tensor = field(repr=False)
    params: Matrix

    def __post_init__(self):
        if isinstance(self.image, np.ndarray):
            print("convert to tensor")
            self.image = tf.convert_to_tensor(self.image)
            self.image = self.image[:, :, :3]
        elif isinstance(self.image, (bytes, bytearray)):
            self.image = ImageManager.load_image(self.image)
            self.image = self.image[:, :, :3]
        if not (self.image.shape[0] == 1 or self.image.ndim == 4):
            self.image = tf.reshape(
                self.image, (-1, self.image.shape[0], self.image.shape[1], 3)
            )

    @property
    def np_image(self) -> np.ndarray:
        """
        Returns the image as a numpy array.
        """
        return self.image.numpy()[0]
