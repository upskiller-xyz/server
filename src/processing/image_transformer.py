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
import cv2
from math import degrees, ceil
import logging
import numpy as np
import tensorflow as tf

from .coord import Coord
from .colorconverter import ColorConverter
from .utils import TARGET_SIZE, OUTPUT_SIZE

logger = logging.getLogger(__name__)


class SceneSettings:
    width: float = 12800.0
    height: float = 12800.0

    @classmethod
    def get_scale(cls, axis: int = 0) -> float:
        if axis == 0:
            return TARGET_SIZE / cls.width
        return TARGET_SIZE / cls.height

    @classmethod
    def to_pixels(cls, coord: Coord) -> Coord:
        """
        Converts coordinates from meters to pixels.
        :param coord: coordinates in meters, Coord
        :return: coordinates in pixels, Coord
        """
        return Coord(int(coord.x * cls.get_scale(0)), int(coord.y * cls.get_scale(1)))


class ImageTransformer:
    """
    Class that handles image transformations needed for the single predictions combination.
    """

    # _colorscale = ColorScale.from_cloud()

    @classmethod
    def align(cls, img: np.ndarray, trans: Coord) -> np.ndarray:
        img = cls._to_zero(img)
        return cls.translate(img, trans)

    @classmethod
    def color_convert(cls, img: np.ndarray) -> np.ndarray:
        """
        Color postprocessing for target images. Assigns standard colorscale values to the analysis results image.
        :param: img       image to cluster the colors of, tf.Tensor
        returns:          image with clustered colors, tf.Tensor
        """
        # if not cls._cnv:
        #    cls._cnv.init()
        res = ColorConverter.make(img)
        return cls.mask(res)

    @classmethod
    def combine(cls, imgs: np.ndarray) -> np.ndarray:
        """
        Combines multiple images into one by summing their pixel values.
        :param imgs: list of images to combine, np.ndarray
        :return: combined image, np.ndarray
        """
        _mask = cls.get_mask(imgs[0])
        imgs = [cls.mask(img) for img in imgs]
        if isinstance(imgs, list):
            imgs = np.array(imgs)
        labels = [ColorConverter.get_values(img) for img in imgs]
        labels = np.sum(np.stack(labels, axis=0), axis=0)
        res = ColorConverter.values_to_image(labels)
        res[~_mask] = [255, 255, 255]
        return res

    @classmethod
    def _remove_black_line(cls, img: np.ndarray) -> np.ndarray:
        """
        Removes black line from the image.
        :param img: image to remove black line from, np.ndarray
        :return: image without black line, np.ndarray
        """
        if img.shape[0] == 1 or img.shape[1] == 1:
            return img
        if np.all(img[0, :] == 0):
            img[0, :] = [255, 255, 255]
        if np.all(img[:, 0] == 0):
            img[:, 0] = [255, 255, 255]
        return img

    @classmethod
    def coords(cls, image: np.ndarray) -> tuple[int, int]:
        image = cls._remove_black_line(image)
        _vs = np.where(image < 1)
        y = np.min(_vs[0])
        x = np.min(_vs[1])
        return Coord(x, y)

    @classmethod
    def denorm(cls, img: tf.Tensor) -> tf.Tensor:
        return (img + 1) * 127.5

    @classmethod
    def get_mask(cls, img: np.ndarray) -> np.ndarray:
        """
        Condiders the image as a mask if it has any pixel with value less than 250.
        """
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        return (img < 250).any(axis=2)

    @classmethod
    def mask(cls, img: np.ndarray) -> np.ndarray:
        """
        Returns a masked image where pixels with value less than 250 are kept from the original input.
        :param img: image to get the mask from, np.ndarray
        :return: mask of the image, np.ndarray
        """
        img[~cls.get_mask(img)] = 0.0
        return img

    @classmethod
    def norm(cls, img: tf.Tensor) -> tf.Tensor:
        return (img / 127.5) - 1

    @classmethod
    def resize(
        cls, img: np.ndarray, h: int = OUTPUT_SIZE, w: int = OUTPUT_SIZE
    ) -> np.ndarray:
        """
        Resize image to the target size.
        :param img: image to resize, np.ndarray
        :param h: target height, int
        :param w: target width, int
        :return: resized image, np.ndarray
        """
        if hasattr(img, "numpy") and len(img.shape) == 4:
            img = img.numpy()[0]
        if img.shape[0] == h and img.shape[1] == w:
            return img
        logger.debug(
            "Resizing {} from shape {} to {}".format(type(img), img.shape, (w, h))
        )
        res = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST_EXACT)
        if res.ndim == 2:  # grayscale fallback
            res = res[..., np.newaxis]
        return res

    @classmethod
    def rotate(cls, image: np.ndarray, angle: float):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(
            image, rot_mat, image.shape[1::-1], flags=cv2.INTER_NEAREST
        )

        return result

    @classmethod
    def run(
        cls, image: np.ndarray, angle: float, cs: Coord, id:int=0
    ) -> np.ndarray:  # inp:PredictionInput)->PredictionInput:
        angle = ceil(degrees(angle))
        res = cls.rotate(image, angle)
        res = cls.align(res, cs)
        return res

    @classmethod
    def translate(cls, img: np.ndarray, trans: Coord) -> np.ndarray:
        return np.roll(img, (int(trans.y), int(trans.x)), axis=(0, 1))

    @classmethod
    def _to_zero(cls, img: np.ndarray) -> np.ndarray:
        _cs = cls.coords(img)
        return cls.translate(img, Coord(-_cs.x, -_cs.y))
