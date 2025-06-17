from __future__ import annotations
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import cv2
import os

from .prediction_input import PredictionInput

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