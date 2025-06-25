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
import base64
import cv2
import numpy as np
import io

import logging

logger = logging.Logger(__name__)


class Encoder:

    @classmethod
    def str_to_bytes(cls, bytestring: str) -> bytes:
        if isinstance(bytestring, str):
            return base64.b64decode(bytestring)
        return bytestring

    @classmethod
    def np_to_base64(cls, matrix: np.array) -> str:
        # return base64.b64encode(matrix.tobytes()).decode('utf-8')
        buf = io.BytesIO()
        np.save(buf, matrix)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    @classmethod
    def base64_to_np_arr(cls, b64str: str) -> np.array:
        # If input is bytes, decode to str
        if isinstance(b64str, bytes):
            b64str = b64str.decode("utf-8")
        arr_bytes = base64.b64decode(b64str)
        print("Decoded base64 length:", len(arr_bytes))
        arr = np.load(io.BytesIO(arr_bytes))
        return arr

    @classmethod
    def base64_to_np(cls, matrix: str) -> np.array:
        try:
            matrix = cls.str_to_bytes(matrix)
            img = np.frombuffer(matrix, dtype=np.uint8)
            return cv2.imdecode(img, flags=1)
        except Exception as e:
            logger.exception(e)
            return cls.base64_to_np_arr(matrix)
