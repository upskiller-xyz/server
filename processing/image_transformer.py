from __future__ import annotations
from dataclasses import dataclass
import cv2
from math import degrees
import logging
import numpy as np
import tensorflow as tf

from .prediction_input import PredictionInput
from .colorscale import ColorScale
from .colorconverter import ColorConverter
from .utils import TARGET_SIZE

class SceneSettings:
    width:float = 12800.0
    height:float = 12800.0
    
    @classmethod
    def get_scale(cls, axis:int=0)->float:
        if axis == 0:
            return TARGET_SIZE / cls.width 
        return TARGET_SIZE / cls.height 


class ImageTransformer:
    """
    Class that handles image transformations needed for the single predictions combination.
    """
    _colorscale = ColorScale.from_cloud()
    _cnv = ColorConverter(_colorscale)

    @classmethod
    def run(cls, inp:PredictionInput)->PredictionInput:
        angle = degrees(inp.params.rotation)
        transl = inp.params.translation
        res = cls.rotate(inp.image.numpy(), angle)
        res = cls.translate(res, transl)
        return PredictionInput(inp.id, res, inp.params)
    
    @classmethod
    def mask(cls, img: np.ndarray)->np.ndarray:
        """
        Condiders the image as a mask if it has any pixel with value less than 250.
        """
        return (img < 250).any(axis=2)
    
    @classmethod
    def rotate(cls, img: np.ndarray, angle: float)->np.ndarray:
        return cv2.rotate(img, angle)
    
    @classmethod
    def translate(cls, img:np.ndarray, trans: tuple[float])->np.ndarray:
        
        x = -1 * trans[0] * SceneSettings.get_scale(0)
        y = -1 * trans[1] * SceneSettings.get_scale(1)
        
        return np.roll(img, (y, x), axis=(0, 1))
    
    @classmethod
    def color_convert(cls, img:np.ndarray)->np.ndarray:
        """
        Color postprocessing for target images. Assigns standard colorscale values to the analysis results image.
        :param: img       image to cluster the colors of, tf.Tensor
        returns:          image with clustered colors, tf.Tensor
        """
        # if not cls._cnv:
        #    cls._cnv.init()
        res = cls._cnv.make(img)
        res[~cls.mask(img)] = 0.0 
        return res
    @classmethod
    def combine(cls, imgs:np.ndarray)->np.ndarray:
        return np.sum(np.stack(imgs, axis=0), axis=0)



