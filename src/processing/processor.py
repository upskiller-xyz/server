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
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

# import cv2
import os

# from .prediction_input import PredictionInput


class MultiThreader:

    num_workers = 4  # Default number of workers

    @classmethod
    def estimate_workers(cls, num_inputs: int = 1) -> int:
        cls.num_workers = int(
            np.min([np.max([1, num_inputs]), (os.cpu_count() or 1) * 5])
        )
        return cls.num_workers

    @classmethod
    def run(cls, func: exec, inp):
        futures = cls._run(func, inp)
        # Yield results as they complete
        return [
            f.result() for f in futures
        ]  # (f.result() for f in as_completed(futures))

    @classmethod
    def _run(cls, func: exec, inp=[]):
        _ = cls.estimate_workers(len(inp))
        with ThreadPoolExecutor(max_workers=cls.num_workers) as executor:
            return [executor.submit(func, i) for i in inp]


#############
# 1. Receive images, rotation, translation
# 2. Get predictions for each image
# 3. combine images
# 4. translate and rotate
# 5. send back
