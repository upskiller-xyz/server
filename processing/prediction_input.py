from __future__ import annotations
from dataclasses import dataclass
import logging
import tensorflow as tf

@dataclass
class Matrix:
    rotation: float
    translation: float

@dataclass
class PredictionInput:
    id: str
    image: tf.Tensor
    params: Matrix
        
