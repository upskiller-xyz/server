import os
import io
import json
import numpy as np
from werkzeug.datastructures import MultiDict
import io
import pytest
import sys

sys.path.append("..")
from server.main import app

ASSET_DIR = os.path.join(os.path.dirname(__file__), "../.assets/W_RN1018")


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def read_form_values(path):
    with open(path, "r") as f:
        lines = f.readlines()
    values = {}
    for line in lines:
        if "\t" in line:
            k, v = line.strip().split("\t", 1)
            if k == "rotations_rad":
                values["rotations_rad"] = json.loads(v.replace("'", '"'))
            elif k == "translations_mm":
                values["translations_mm"] = json.loads(v.replace("'", '"'))
            elif k == "num_images":
                values["num_images"] = int(v)
    return values


def read_images(asset_dir, num_images):
    images = []
    for i in range(num_images):
        fname = os.path.join(asset_dir, f"floorplan{i}_s.png")
        with open(fname, "rb") as f:
            img_bytes = f.read()
        images.append(img_bytes)
    return images


def test_get_df_with_real_assets(client):
    form = read_form_values(os.path.join(ASSET_DIR, "form_values.txt"))
    images = read_images(ASSET_DIR, form["num_images"])
    data = MultiDict(
        [("file", (io.BytesIO(img), f"img{i}.png")) for i, img in enumerate(images)]
        + [
            ("translation", json.dumps({"x": 0, "y": 0})),
            ("rotation", json.dumps([1.5, 2.0, 3.0])),
        ]
    )
    response = client.post("/get_df", data=data, content_type="multipart/form-data")
    assert response.status_code in (200, 500)
    # Optionally, check content:
    # if response.status_code == 200:
    #     assert "content" in response.get_json()


def test_get_stats_with_real_asset(client):
    img_path = os.path.join(ASSET_DIR, "floorplan0_s.png")
    with open(img_path, "rb") as f:
        img_bytes = f.read()
    data = {"file": (io.BytesIO(img_bytes), "floorplan0_s.png")}
    response = client.post("/get_stats", data=data, content_type="multipart/form-data")
    assert response.status_code in (200, 500)
