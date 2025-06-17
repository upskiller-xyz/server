import numpy as np
import tensorflow as tf
import pytest
from unittest import mock

from ..processing.image_manager import ImageManager

def test_norm_and_denorm():
    img = tf.constant([[0.0, 127.5, 255.0]])
    normed = ImageManager._norm(img)
    denormed = ImageManager.denorm(normed)
    tf.debugging.assert_near(denormed, img)

def test_resize():
    img = tf.ones((10, 10, 3))
    out = ImageManager._resize(img, 5, 5)
    assert out.shape == (5, 5, 3)

def test_process_image_norm():
    img = tf.ones((32, 32, 3)) * 255
    out = ImageManager._process_image(img, 16, 16, norm=True)
    assert out.shape == (16, 16, 3)
    assert tf.reduce_max(out) <= 1.0

def test_process_image_no_norm():
    img = tf.ones((32, 32, 3)) * 255
    out = ImageManager._process_image(img, 16, 16, norm=False)
    assert out.shape == (16, 16, 3)
    assert tf.reduce_max(out) == 255.0

def test_save_success(tmp_path):
    img = tf.ones((8, 8, 3), dtype=tf.uint8)
    fname = str(tmp_path / "test.png")
    with mock.patch("tensorflow.io.write_file") as write_file, \
         mock.patch("tensorflow.image.encode_png", return_value=b"abc"):
        write_file.return_value = None
        assert ImageManager.save(img, fname) is True

def test_save_failure():
    img = tf.ones((8, 8, 3), dtype=tf.uint8)
    with mock.patch("tensorflow.io.write_file", side_effect=Exception("fail")), \
         mock.patch("tensorflow.image.encode_png", return_value=b"abc"):
        assert ImageManager.save(img, "badfile.png") is False

def test_load_image(monkeypatch):
    dummy_bytes = b"123"
    dummy_img = np.ones((8, 8, 3), dtype=np.uint8)
    monkeypatch.setattr("cv2.imdecode", lambda arr, flags: dummy_img)
    out = ImageManager.load_image(dummy_bytes)
    assert isinstance(out, tf.Tensor)
    assert out.shape == (8, 8, 3)

def test_get_image(monkeypatch):
    dummy_img = tf.ones((8, 8, 3))
    monkeypatch.setattr(ImageManager, "load_image", lambda b: dummy_img)
    monkeypatch.setattr(ImageManager, "_process_image", lambda img, h, w, norm: img + 1)
    out = ImageManager.get_image(b"abc", 8, 8, True)
    assert tf.reduce_all(out == dummy_img + 1)