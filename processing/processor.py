from __future__ import annotations
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import cv2
import os

@dataclass
class Matrix:
    rotation: float
    translation: float

@dataclass
class PredictionInput:
    text: str
    model_name: str
    parameters: Matrix

    # def __post_init__(self):
    #     if not isinstance(self.text, str):
    #         raise ValueError("text must be a string")
    #     if not isinstance(self.model_name, str):
    #         raise ValueError("model_name must be a string")
    #     if not isinstance(self.parameters, dict):
    #         raise ValueError("parameters must be a dictionary")

    @classmethod
    def load_image(cls, bytesting)->np.ndarray:
        """
        Load a PredictionInput instance from a testing source.
        This is a placeholder for actual loading logic.
        """
        img = np.frombuffer(bytesting, dtype=np.uint8)
        return cv2.imdecode(img, flags=1)
        

class MultiThreader:
    
    num_workers = 4  # Default number of workers

    @classmethod
    def estimate_workers(cls, num_inputs:int=1)->int:
        cls.num_workers = int(np.min(np.max(1, num_inputs), (os.cpu_count() or 1) * 5))
        return cls.num_workers

    @classmethod
    def run(cls, func:exec, inp):
        return as_completed([x.result() for x in cls._run(func, inp)])
    
    @classmethod
    def _run(cls, func:exec, inp=[]):
        _ = cls.estimate_workers(len(inp))
        with ThreadPoolExecutor(max_workers=cls.num_workers) as executor:
            return [executor.submit(func, i) for i in inp]


class Processor:

    @classmethod
    def run(cls, inp: PredictionInput)->None:
        return cls._run(inp)
    
    @classmethod
    def _run(cls, inp: PredictionInput) -> None:
        raise NotImplementedError("Subclasses must implement this method")
    



#############
# 1. Receive images, rotation, translation
# 2. Get predictions for each image
# 3. combine images
# 4. translate and rotate
# 5. send back