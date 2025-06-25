# Daylight Factor Simulation Server
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
import numpy as np


from .colorscale import ColorScale


class ColorConverter:
    colorscale: ColorScale = ColorScale.from_cloud()

    @classmethod
    def make(cls, img: np.ndarray) -> np.ndarray:
        """
        Function to convert the image to the colorscale values
        :param img: image to convert to colorscale values in RGB, np.array
        :return: image with converted colors in RGB, np.array

        """
        colors = np.array([list(x) for x in cls.colorscale.colors_lab])
        labels = cls.solve(img)
        return colors[labels]

    @classmethod
    def to_lab(cls, img: np.ndarray) -> np.ndarray:

        return cv2.cvtColor(np.array(img).astype(np.uint8)[:, :, :3], cv2.COLOR_RGB2LAB)

    @classmethod
    def solve(cls, img: np.array) -> np.array:
        """
        Function to get the labels of the colorscale values for the image
        :param img: image to get the labels for in RGB, np.array
        :return: labels of the colorscale values, np.array
        """
        img = cls.to_lab(img)
        colors = np.array([list(x) for x in cls.colorscale.colors_lab])
        color_distance = img[:, :, np.newaxis, :].astype(np.int16) - colors.reshape(
            1, 1, -1, 3
        ).astype(np.int16)
        labels = np.linalg.norm(color_distance, axis=3).argmin(axis=2)
        return labels

    @classmethod
    def get_values(cls, img: np.array) -> np.array:
        values = np.array([x.value for x in cls.colorscale.colors])
        labels = cls.solve(img)
        return values[labels]

    @classmethod
    def label(cls, img: np.array) -> np.array:
        """
        Function to label the image with the colorscale values
        :param img: image to label in RGB, np.array
        :return: labeled image, np.array
        """
        return cls.solve(img)

    @classmethod
    def _make(cls, img: np.array) -> np.array:
        """
        Function to convert the image to the colorscale values
        :param img: image to convert to colorscale values in RGB, np.array
        :return: image with converted colors in RGB, np.array
        """
        if not cls.clustering:
            cls.clustering = cls.init()
        res = cls.cluster(img)
        out_labels = res.labels_.reshape(*img.shape[:2])
        return np.array([list(x.rgb) for x in cls.colorscale.colors])[out_labels]

    @classmethod
    def values_to_image(cls, labels: np.array) -> np.array:
        if labels.ndim > 2:
            labels = labels[:, :, 0]
        scale_colors = cls.colorscale.value_to_color(labels)

        return (
            np.array([list(x.rgb) for x in scale_colors.ravel()])
            .reshape(*labels.shape[:2], 3)
            .astype(np.uint8)
        )
