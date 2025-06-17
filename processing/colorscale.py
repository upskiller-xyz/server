from __future__ import annotations
from dataclasses import dataclass, field
import cv2
import numpy as np
from enum import Enum
import json

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
    r:int
    g:int
    b:int
    value:float

    @property
    def rgb(self) -> tuple[int, int, int]:
        """
        Color's formatted rgb representation.
        """
        return (self.r, self.g, self.b)
    
    @classmethod
    def background(cls)->ScaleColor:
        """
        Method generating a background color with the value assigned as -1.
        """
        return ScaleColor(0,0,0,-1)

@dataclass(frozen=True)
class ColorScale:
    """
    Class containing the analysis colorscale. Keeps colors with their assigned values.
    """
    colors: list[ScaleColor] = field(default_factory=list)

    @property
    def colors_lab(self) -> list[tuple[int, int, int]]:
        # return rgb2lab([color.rgb for color in self.colors])
        return cv2.cvtColor(np.array([[list(x.rgb) for x in self.colors]]).astype(np.uint8)[:,:,:3], cv2.COLOR_RGB2Lab)[0]

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
    def from_cloud(cls, cs:COLORSCALES=COLORSCALES.DF) -> ColorScale:
        """
        Method that loads a certain colorscale from GCP.
        """
        json_str = cls.load(cs)
        if json_str:
            return cls.from_json(json_str)
        return None

    @classmethod
    def load(cls, cs:COLORSCALES=COLORSCALES.DF) -> ColorScale:
        """
        Method that loads a certain colorscale from GCP.
        """
        try:
            return GCSManager.load(cs.value, bucket_name=ASSET_BUCKET)
        except Exception as e:
            print(e)
            return None
