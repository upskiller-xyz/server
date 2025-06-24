import cv2
import numpy as np
import tensorflow as tf
import pytest

from ..processing.image_transformer import ImageTransformer
from ..processing.image_manager import ImageManager
from ..processing.prediction_input import PredictionInput, Matrix
from ..processing.inference import Inference
from ..processing.steps.step import ImagePreprocessStep, PredictStep
from ..processing.utils import TARGET_SIZE


def test_image_manager_get_image_shape_and_type():
    # Create a dummy PNG image
    img = np.ones((64, 64, 3), dtype=np.uint8) * 127
    _, img_encoded = cv2.imencode(".png", img)
    img_bytes = img_encoded.tobytes()
    tensor = ImageManager.get_image(img_bytes, h=128, w=128)
    assert isinstance(tensor, tf.Tensor)
    assert tensor.shape == (128, 128, 3)
    assert tensor.dtype == tf.float32


def test_prediction_input_converts_numpy_to_tensor():
    arr = np.ones((32, 32, 3), dtype=np.float32)
    pred = PredictionInput("id", arr, Matrix(0, (0, 0, 0)))
    assert isinstance(pred.image, tf.Tensor)
    assert pred.image.shape == (1, 32, 32, 3)


def test_prediction_input_converts_bytes_to_tensor():
    img = np.ones((32, 32, 3), dtype=np.uint8) * 200
    _, img_encoded = cv2.imencode(".png", img)
    img_bytes = img_encoded.tobytes()
    pred = PredictionInput("id", img_bytes, Matrix(0, (0, 0, 0)))
    assert isinstance(pred.image, tf.Tensor)
    assert pred.image.shape[3] == 3


def test_image_manager_normalization():
    img = tf.ones((16, 16, 3), dtype=tf.float32) * 127.5
    normed = ImageTransformer.norm(img)
    assert tf.reduce_max(normed) <= 1.0
    assert tf.reduce_min(normed) >= -1.0


def test_image_manager_denormalization():
    img = tf.ones((16, 16, 3), dtype=tf.float32)
    normed = ImageTransformer.norm(img)
    denormed = ImageTransformer.denorm(normed)
    assert np.allclose(denormed.numpy(), img.numpy())


def test_inference_input_shape_and_normalization(monkeypatch):
    # Patch model to just return the input
    monkeypatch.setattr(Inference, "model", lambda x, training=False: x)
    pred = PredictionInput(
        "id", tf.ones((TARGET_SIZE, TARGET_SIZE, 3)), Matrix(0, (0, 0, 0))
    )
    out = Inference.run(pred)
    assert isinstance(out, PredictionInput)
    assert out.image.shape == (1, TARGET_SIZE, TARGET_SIZE, 3) or out.image.shape == (
        TARGET_SIZE,
        TARGET_SIZE,
        3,
    )
    # Check normalization range
    assert tf.reduce_max(ImageTransformer.norm(out.image)) <= 1.0
    assert tf.reduce_min(ImageTransformer.norm(out.image)) >= -1.0


def test_image_preprocess_step_output_shape_and_type():
    pred = PredictionInput("id", tf.ones((64, 64, 3)), Matrix(0, (0, 0, 0)))
    inp = type("Dummy", (), {"value": pred})()
    out = ImagePreprocessStep._run(inp)
    assert out.value.image.shape == (1, TARGET_SIZE, TARGET_SIZE, 3)
    assert isinstance(out.value.image, tf.Tensor)
