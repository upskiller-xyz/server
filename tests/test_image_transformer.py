import cv2
import numpy as np
import pytest
import tensorflow as tf

from ..processing.image_transformer import ImageTransformer, Coord


def make_test_img(c1, c2, value=0.0, bg=255.0):
    # Create a white 10x10 image with a square of 'value' at (c1:c2, c1:c2)
    img = np.ones((10, 10, 3), dtype=np.float32) * bg
    img[c1:c2, c1:c2] = value
    return img


def test_remove_black_line():
    img = np.zeros((10, 10, 3), dtype=np.float32)
    img[1:, 1:] = 255
    out = ImageTransformer._remove_black_line(img.copy())
    assert np.all(out[0, :] == 255)
    # assert np.all(out[:, 0] == 255)


def test_coords_finds_top_left():
    img = make_test_img(5, 7, value=0.0, bg=255.0)
    coord = ImageTransformer.coords(img)
    assert coord.x == 5 and coord.y == 5


def test_to_zero_moves_content_to_top_left():
    img = make_test_img(5, 7, value=0.0, bg=255.0)
    img_zeroed = ImageTransformer._to_zero(img.copy())
    # The black square should now be at (0,0)
    assert np.all(img_zeroed[0:2, 0:2] == 0)
    # The rest should be white
    assert np.all(img_zeroed[2:, :, :] == 255)
    assert np.all(img_zeroed[:, 2:, :] == 255)


def test_translate_moves_content():
    img = make_test_img(0, 2, value=0.0, bg=255.0)
    # Move by (3,4): right 3, down 4
    translated = ImageTransformer.translate(img, Coord(3, 4))
    # The black square should now be at (4:6, 3:5)
    assert np.all(translated[4:6, 3:5] == 0)
    # The original top-left should now be white
    assert np.all(translated[0:4, :, :] == 255)
    assert np.all(translated[:, 0:3, :] == 255)


def test_norm_and_denorm():
    img = tf.constant([[0.0, 127.5, 255.0]])
    normed = ImageTransformer.norm(img)
    denormed = ImageTransformer.denorm(normed)
    tf.debugging.assert_near(denormed, img)


def test_resize():
    a = np.ones((10, 10, 3), dtype=np.float32)
    out = ImageTransformer.resize(a, 5, 5)
    assert out.shape == (5, 5, 3), "{} -> {} : {}".format(a.shape, out.shape, (5, 5, 3))


def test_get_mask_and_mask():
    img = np.ones((4, 4, 3), dtype=np.float32) * 255
    img[0, 0, :] = 100
    mask = ImageTransformer.get_mask(img)
    assert mask.shape == (4, 4)
    assert mask[0, 0]
    assert not mask[1, 1]
    masked = ImageTransformer.mask(img.copy())
    assert np.all(masked[~mask] == 0)


def test_rotate_identity():
    img = np.eye(4, 4, 3).astype(np.float32)
    out = ImageTransformer.rotate(img, 0)
    assert out.shape == img.shape


def test_run(monkeypatch, tmp_path):
    img = np.ones((8, 8, 3), dtype=np.float32) * 100
    angle = 0.5
    cs = Coord(1, 2)
    # Patch cv2.imwrite to avoid file output
    # monkeypatch.setattr("cv2.imwrite", lambda *a, **k: None)
    out = ImageTransformer.run(img, angle, cs)
    assert out.shape == img.shape
