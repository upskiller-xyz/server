import os
import base64
import numpy as np
import tensorflow as tf
import pytest

from ..processing.pipeline import GetDfPipeline
from ..processing.pipelineinput import GetDfPipelineInput

ASSET_DIR = os.path.join(os.path.dirname(__file__), "../.assets/W_RN1034")


def read_form_values(path):
    with open(path, "r") as f:
        lines = f.readlines()
    values = {}
    for line in lines:
        if "\t" in line:
            k, v = line.strip().split("\t", 1)
            if k == "rotations_rad":
                values["rotations_rad"] = eval(v)
            elif k == "translations_mm":
                values["translations_mm"] = eval(v)
            elif k == "num_images":
                values["num_images"] = int(v)
    return values


def read_images(asset_dir, num_images):
    images = []
    for i in range(num_images):
        fname = os.path.join(asset_dir, f"floorplan{i}.png")
        with open(fname, "rb") as f:
            img_bytes = f.read()
        # img_b64 = base64.b64encode(img_bytes)  #.decode("utf-8")
        images.append(img_bytes)
    return images


# def test_get_df_pipeline_on_asset(monkeypatch):
#     # Read form values and images
#     form_path = os.path.join(ASSET_DIR, "form_values.txt")
#     form = read_form_values(form_path)
#     images_b64 = read_images(ASSET_DIR, form["num_images"])

#     # Patch ImageManager.get_image to decode base64 and return a dummy tensor
#     def fake_get_image(b64, h=128, w=128, norm=True):
#         # In real test, decode and process, here just return a tensor
#         return tf.ones((128, 128, 3), dtype=tf.float32)
#     monkeypatch.setattr("server.processing.image_manager.ImageManager.get_image", fake_get_image)

#     # Build pipeline input
#     pipeline_input = GetDfPipelineInput.build(images_b64)
#     if isinstance(pipeline_input, list):
#         pipeline_input = GetDfPipelineInput(pipeline_input)

#     # Run the pipeline
#     result = GetDfPipeline.run(pipeline_input)
#     assert result is not None

#     # --- Assertions ---
#     # If result is a PipelineInput, get the tensor
#     img = result.value if hasattr(result, "value") else result
#     if isinstance(img, list):
#         img = img[0]
#     if hasattr(img, "numpy"):
#         img = img.numpy()
#     assert img.shape == (128, 128, 3)
#     # Not all white
#     assert not np.allclose(img, 255)
#     # Not all black
#     assert not np.allclose(img, 0)

# def test_get_df_pipeline_on_asset_real():
#     form_path = os.path.join(ASSET_DIR, "form_values.txt")
#     form = read_form_values(form_path)
#     images_b64 = read_images(ASSET_DIR, form["num_images"])
#     pipeline_input = GetDfPipelineInput.build(images_b64)
#     if isinstance(pipeline_input, list):
#         pipeline_input = GetDfPipelineInput(pipeline_input)
#     result = GetDfPipeline.run(pipeline_input)
#     assert result is not None

#     # --- Assertions ---
#     img = result.value if hasattr(result, "value") else result
#     if isinstance(img, list):
#         img = img[0]
#     if hasattr(img, "numpy"):
#         img = img.numpy()
#     assert img.shape == (128, 128, 3)
#     assert not np.allclose(img, 255)
#     assert not np.allclose(img, 0)
