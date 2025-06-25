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
import cv2
import numpy as np
from enum import Enum
import json
import logging

logger = logging.Logger(__name__)

from .gcs_manager import GCSManager

ASSET_BUCKET = "daylight_analysis_assets"


class COLORSCALES(Enum):
    """
    All the colorscales used for our analyses, with their paths to a protected bucket on GCP.
    """

    DF = "colorscale_df.json"
    DA = "colorscale_da.json"


@dataclass(frozen=True)
class ScaleColor:
    """
    Color from a colorscale with its assigned value.
    """

    r: int
    g: int
    b: int
    value: float

    @property
    def rgb(self) -> tuple[int, int, int]:
        """
        Color's formatted rgb representation.
        """
        return (self.r, self.g, self.b)

    @classmethod
    def background(cls) -> ScaleColor:
        """
        Method generating a background color with the value assigned as -1.
        """
        return ScaleColor(0, 0, 0, -1)


@dataclass(frozen=True)
class ColorScale:
    """
    Class containing the analysis colorscale. Keeps colors with their assigned values.
    """

    colors: list[ScaleColor] = field(default_factory=list)

    @property
    def colors_lab(self) -> list[tuple[int, int, int]]:
        # return rgb2lab([color.rgb for color in self.colors])
        return cv2.cvtColor(
            np.array([[list(x.rgb) for x in self.colors]]).astype(np.uint8)[:, :, :3],
            cv2.COLOR_RGB2LAB,
        )[0]

    def to_json(self) -> str:
        return json.dumps(self, default=lambda o: o.__dict__, indent=4)

    @staticmethod
    def from_json(json_str: str) -> ColorScale:
        """
        Method that constructs colorscale from a json string. Json in the string is expected to have the following format: [{"Color": [0,0,0], "Value": 0}]
        """
        d = json.loads(json_str)
        colors = [ScaleColor(*color["Color"], color["Value"]) for color in d]
        colors.append(ScaleColor.background())
        return ColorScale(colors=colors)

    @classmethod
    def from_cloud(cls, cs: COLORSCALES = COLORSCALES.DF) -> ColorScale:
        """
        Method that loads a certain colorscale from GCP.
        """
        json_str = cls.load(cs)
        if json_str:
            return cls.from_json(json_str)
        return None

    @classmethod
    def load(cls, cs: COLORSCALES = COLORSCALES.DF) -> ColorScale:
        """
        Method that loads a certain colorscale from GCP.
        """
        try:
            return GCSManager.load(cs.value, bucket_name=ASSET_BUCKET)
        except Exception as e:
            logger.exception(e)
            return None

    def value_to_color(self, value: np.array) -> np.array[ScaleColor]:
        """
        Method that gets the closest color to the given label.
        """
        dist = np.array([x.value for x in self.colors])[:, np.newaxis] - np.array(
            value
        ).reshape(1, 1, -1)
        idx = np.abs(dist).argmin(axis=1)[0]

        return np.array(self.colors)[idx]
