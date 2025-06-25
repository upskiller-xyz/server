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
import base64
import logging
import numpy as np
import tensorflow as tf

from .encoder import Encoder

logger = logging.getLogger(__name__)

from .image_transformer import ImageTransformer
from .utils import TARGET_SIZE


@dataclass(frozen=True)
class ImgInput:
    inp: any
    target: any

    @property
    def out(self) -> tuple[any, any]:
        return (self.inp, self.target)


@dataclass(frozen=True)
class ImgFname:
    inp: str
    target: str

    @property
    def out(self) -> tuple[str, str]:
        return (self.inp, self.target)


class ImageManager:

    @classmethod
    def save(cls, img: tf.Tensor, fname: str) -> bool:
        """
        Save image to disk
        :param img: image to save, tf.Tensor
        :param fname: filename to save to, str
        :return: True if successful, False otherwise
        """
        try:
            img = tf.cast(img, tf.uint8)
            tf.io.write_file(fname, tf.image.encode_png(img))
            return True
        except Exception as e:
            logging.error(f"Error saving image: {e}")
            return False

    @classmethod
    def get_image(
        cls,
        bytestring: str,
        h: int = TARGET_SIZE,
        w: int = TARGET_SIZE,
        norm: bool = True,
    ) -> tf.Tensor:
        image = cls.load_image(bytestring)
        return cls._process_image(image, h, w, norm)

    @classmethod
    def _load_image(cls, bytestring: str) -> tf.Tensor:
        img = Encoder.base64_to_np(bytestring)
        return tf.cast(img[:, :, :3], tf.float32)

    @classmethod
    def load_image(cls, img_bytes: bytes) -> tf.Tensor:
        """
        Load an image from a base64 string.
        :param img_b64: The base64 string of the image.
        :return: The image as a tf.Tensor.
        """

        # img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        # i = base64.b64decode(img_b64)
        i = tf.io.decode_image(img_bytes)
        i = tf.cast(i[:, :, :3], tf.float32)
        return ImageTransformer.norm(i)

    @classmethod
    def encode(cls, img: tf.Tensor) -> str:
        """
        Encode an image to a base64 string.
        :param img: The image as a tf.Tensor.
        :return: The base64 string of the image.
        """
        img = tf.cast(img, tf.uint8)
        encoded_img = tf.image.encode_png(img)
        b64 = tf.io.encode_base64(encoded_img).numpy()
        b64 = base64.b64encode(encoded_img)
        return b64.decode("utf-8")

    @classmethod
    def _process_image(
        cls,
        img: tf.Tensor,
        h: int = TARGET_SIZE,
        w: int = TARGET_SIZE,
        norm: bool = True,
    ) -> tf.Tensor:
        img = tf.cast(img[:, :, :3], tf.float32)
        img = cls._resize(img, h, w)
        if norm:
            img = ImageTransformer.norm(img)
        return img

    @classmethod
    def _resize(
        cls, img: tf.Tensor, h: int = TARGET_SIZE, w: int = TARGET_SIZE
    ) -> tf.Tensor:
        return tf.image.resize(
            img, [h, w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
