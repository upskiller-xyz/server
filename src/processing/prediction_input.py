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
