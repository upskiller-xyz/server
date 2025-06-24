import cv2
import numpy as np
import io
import base64
import pytest
from ..processing.encoder import Encoder


def test_np_to_base64_and_back_float():
    arr = np.random.uniform(0, 10, (5, 7)).astype(np.float32)
    b64 = Encoder.np_to_base64(arr)
    arr_bytes = base64.b64decode(b64)
    arr_decoded = np.load(io.BytesIO(arr_bytes))
    assert np.allclose(arr, arr_decoded)


def test_np_to_base64_and_back_uint8_image():
    arr = (np.random.rand(8, 8, 3) * 255).astype(np.uint8)
    # Encode as image
    _, img_encoded = cv2.imencode(".png", arr)
    b64 = base64.b64encode(img_encoded.tobytes()).decode("utf-8")
    # Use Encoder.base64_to_np for image
    img_bytes = base64.b64decode(b64)
    img_np = Encoder.base64_to_np(img_bytes)
    assert img_np.shape[0] == 8 and img_np.shape[1] == 8


def test_base64_to_np_invalid(monkeypatch):
    # Should return None or raise if not a valid image
    with pytest.raises(Exception):
        Encoder.base64_to_np("not_base64_data")
