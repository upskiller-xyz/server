import numpy as np
import pytest
from ..processing.image_transformer import ImageTransformer, Coord


def make_test_img(c1, c2):
    # Create a white 10x10 image with a black 2x2 square at (5,5)
    img = np.ones((10, 10, 3), dtype=np.float32)
    img[c1:c2, c1:c2] = 0
    return img


def test_to_zero_moves_content_to_top_left():
    img = make_test_img(5, 7)
    img_zeroed = ImageTransformer._to_zero(img.copy())
    # The black square should now be at (0,0)

    assert np.all(img_zeroed[0:2, 0:2] == 0), "x: {}, y: {}".format(
        np.where(img_zeroed == 0)[1][0], np.where(img_zeroed == 0)[0][0]
    )
    # The rest should be white
    assert np.all(img_zeroed[2:, :, :] == 1)
    assert np.all(img_zeroed[:, 2:, :] == 1)


def test_to_zero_moves_topleft_content_to_top_left():
    img = make_test_img(0, 2)
    img_zeroed = ImageTransformer._to_zero(img.copy())
    # The black square should now be at (0,0)

    assert np.all(img_zeroed[0:2, 0:2] == 0), "x: {}, y: {}".format(
        np.where(img_zeroed == 0)[1][0], np.where(img_zeroed == 0)[0][0]
    )
    # The rest should be white
    assert np.all(img_zeroed[2:, :, :] == 1)
    assert np.all(img_zeroed[:, 2:, :] == 1)


def test_translate_moves_content():
    img = make_test_img(0, 2)
    # Move by (3,4): right 3, down 4
    translated = ImageTransformer.translate(img, Coord(3, 4))
    # The black square should now be at (4:6, 3:5)
    print(translated)
    assert np.all(translated[4:6, 3:5] == 0), "x: {}, y: {}".format(
        np.where(translated == 0)[1][0], np.where(translated == 0)[0][0]
    )
    # The original top-left should now be white
    assert np.all(translated[0:4, :, :] == 1)
    assert np.all(translated[:, 0:3, :] == 1)
