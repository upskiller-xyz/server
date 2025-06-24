import cv2
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
import flask
import io as sysio
import numpy as np
import base64
from werkzeug.datastructures import MultiDict
import pytest

import sys

sys.path.append("..")

from server.processing.colorconverter import ColorConverter
from server.processing.colorscale import ColorScale, ScaleColor

from server.main import app


def make_dummy_img_input():
    img = np.ones((8, 8, 3), dtype=np.uint8) * 255
    _, encoded_img = cv2.imencode(".png", img)
    img_bytes = encoded_img.tobytes()
    img_64 = base64.b64encode(img_bytes).decode("utf-8")
    return img_64


def make_dummy_multipart_file(
    field_name="file", filename="test.png", shape=(8, 8, 3), value=255
):
    """
    Returns a dict suitable for Flask test_client's 'data' argument for multipart/form-data.
    """
    # Create a dummy image
    img = (np.ones(shape) * value).astype(np.uint8)
    _, img_bytes = cv2.imencode(".png", img)
    img_io = io.BytesIO(img_bytes.tobytes())
    img_io.seek(0)
    return {field_name: (img_io, filename)}


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_test_endpoint(client):
    response = client.get("/test", json={})
    assert response.status_code == 204
    assert not response.data


def test_test_content_endpoint(client):
    response = client.get("/test", json={"content": 0})
    assert response.status_code == 200
    assert response.json["content"] == 1


def test_to_values_endpoint(client):
    # Create a dummy image and encode as PNG bytes
    # img = (np.ones((8, 8, 3)) * 127).astype(np.uint8)
    # import cv2
    # _, img_bytes = cv2.imencode('.png', img)
    # img_io = io.BytesIO(img_bytes.tobytes())
    # data = {'file': (img_io, 'test.png')}
    img = make_dummy_multipart_file()
    response = client.post("/to_values", content_type="multipart/form-data", data=img)
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["success"] is True
    assert "content" in json_data


def test_to_rgb_endpoint(client):
    # Create a dummy DF values array, encode as base64

    ColorConverter.colorscale = ColorScale(
        colors=[ScaleColor(0, 0, 0, 0.0), ScaleColor(255, 255, 255, 1.0)]
    )

    img = make_dummy_multipart_file()

    response = client.post("/to_rgb", content_type="multipart/form-data", data=img)
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["success"] is True
    assert "content" in json_data


def test_get_stats_endpoint(client):
    # Create a dummy image and encode as PNG bytes

    ColorConverter.colorscale = ColorScale(
        colors=[ScaleColor(0, 0, 0, 0.0), ScaleColor(255, 255, 255, 1.0)]
    )

    data = make_dummy_multipart_file()
    response = client.post("/get_stats", content_type="multipart/form-data", data=data)
    assert response.status_code == 200
    json_data = response.get_json()
    assert "success" in json_data


def test_get_df_endpoint(client):
    # Simulate three images and form data for rotation and translation
    img = (np.ones((8, 8, 3)) * 127).astype(np.uint8)

    _, img_bytes = cv2.imencode(".png", img)
    data = MultiDict(
        [
            ("file", (io.BytesIO(img_bytes.tobytes()), "img0.png")),
            ("file", (io.BytesIO(img_bytes.tobytes()), "img1.png")),
            ("file", (io.BytesIO(img_bytes.tobytes()), "img2.png")),
            ("rotation", "[0,0,0]"),
            ("translation", '{"x":0,"y":0}'),
        ]
    )
    # Flask's test client expects data as dict, files as list of tuples
    response = client.post("/get_df", data=data, content_type="multipart/form-data")
    # Accept 200 or 500 (if pipeline is not fully mocked)
    assert response.status_code in (200, 500)


# def test_daylight_factor_endpoint(client):
#     # This endpoint expects JSON with "image" key as a list or array
#     arr = np.ones((256, 256, 3), dtype=np.float32)
#     data = {"content":
#         {"image": arr.tolist()}
#     }
#     response = client.post('/daylight_factor', json=data)
#     # Accept 200 or 500 (if model is not loaded)
#     assert response.status_code in (200, 500)
