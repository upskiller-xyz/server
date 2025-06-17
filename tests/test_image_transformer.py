import numpy as np
import pytest
import tensorflow as tf
from unittest import mock

from ..processing.image_transformer import ImageTransformer, SceneSettings
from ..processing.prediction_input import PredictionInput, Matrix

def make_dummy_input():
    img = tf.ones((8, 8, 3), dtype=tf.float32)
    params = Matrix(rotation=0.5, translation=(1, 2, 0))
    return PredictionInput(id="test", image=img, params=params)

def test_mask():
    img = tf.ones((4, 4, 3), dtype=tf.float32) * 255
    img = img.numpy()
    img[0, 0, 0] = 100
    mask = ImageTransformer.mask(img)
    assert mask.shape == (4, 4)
    assert mask[0, 0] == True
    assert not mask[1, 1]

def test_rotate(monkeypatch):
    img = tf.ones((4, 4, 3), dtype=tf.float32)
    img_np = img.numpy()
    called = {}
    def fake_rotate(i, a):
        called['ok'] = (np.array_equal(i, img_np) and a == 45)
        return i
    monkeypatch.setattr("cv2.rotate", fake_rotate)
    out = ImageTransformer.rotate(img_np, 45)
    assert np.array_equal(out, img_np)
    assert called['ok']

def test_translate(monkeypatch):
    img = tf.reshape(tf.range(16*16*3, dtype=tf.float32), (16, 16, 3))
    img_np = img.numpy()
    monkeypatch.setattr(SceneSettings, "get_scale", lambda axis=0: 1)
    out = ImageTransformer.translate(img_np, (1, 2, 0))
    assert out.shape == img_np.shape
    np.testing.assert_array_equal(out, np.roll(img_np, (-2, -1), axis=(0, 1)))

def test_color_convert(monkeypatch):
    img = tf.ones((4, 4, 3), dtype=tf.float32).numpy()
    dummy_res = np.ones((4, 4, 3)) * 42
    monkeypatch.setattr(ImageTransformer._cnv, "make", lambda x: dummy_res.copy())
    monkeypatch.setattr(ImageTransformer, "mask", lambda x: np.array([[True]*4]*4))
    out = ImageTransformer.color_convert(img)
    assert np.all(out == 42)

def test_combine():
    imgs = [tf.ones((2, 2, 3), dtype=tf.float32).numpy(), (tf.ones((2, 2, 3), dtype=tf.float32)*2).numpy()]
    out = ImageTransformer.combine(imgs)
    assert out.shape == (2, 2, 3)
    assert np.all(out == 3)

def test_run(monkeypatch):
    dummy_img = tf.ones((8, 8, 3), dtype=tf.float32)
    params = Matrix(rotation=0.5, translation=(1, 2, 0))
    inp = PredictionInput(id="test", image=dummy_img, params=params)
    monkeypatch.setattr(ImageTransformer, "rotate", lambda img, angle: img)
    monkeypatch.setattr(ImageTransformer, "translate", lambda img, trans: img + 1)
    out = ImageTransformer.run(inp)
    assert isinstance(out, PredictionInput)
    assert np.all(out.image == dummy_img + 1)
    assert out.id == inp.id
    assert out.params == inp.params

def test_run_with_numpy(monkeypatch):
    # Simulate PredictionInput.image as a NumPy array
    dummy_img = np.ones((8, 8, 3), dtype=np.float32)
    params = Matrix(rotation=0.5, translation=(1, 2, 0))
    inp = PredictionInput(id="test", image=dummy_img, params=params)
    # Patch rotate/translate to just add 1 for test
    monkeypatch.setattr(ImageTransformer, "rotate", lambda img, angle: img)
    monkeypatch.setattr(ImageTransformer, "translate", lambda img, trans: img + 1)
    out = ImageTransformer.run(inp)
    assert isinstance(out, PredictionInput)
    assert np.all(out.image == dummy_img + 1)
    assert out.id == inp.id
    assert out.params == inp.params