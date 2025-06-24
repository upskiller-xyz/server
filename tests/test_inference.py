import pytest

import numpy as np
import tensorflow as tf
import mock
import sys

sys.path.append("..")
from ..processing.coord import Coord
from ..processing.inference import Inference
from ..processing.prediction_input import PredictionInput, Matrix
from ..processing.utils import TARGET_SIZE

# python


@pytest.fixture(autouse=True)
def reset_inference_model():
    Inference.model = None
    yield
    Inference.model = None


def make_dummy_input():
    # Create a dummy PredictionInput with random image data
    img = tf.random.uniform((TARGET_SIZE, TARGET_SIZE, 3))
    return PredictionInput(id="test_id", image=img, params=Matrix(0, Coord(0, 0)))


def test_init_success(monkeypatch):
    dummy_model = object()
    monkeypatch.setattr(tf.keras.models, "load_model", lambda *a, **kw: dummy_model)
    assert Inference.init("dummy_path") is True
    assert Inference.model is dummy_model


def test_init_failure(monkeypatch):
    def raise_exc(*a, **kw):
        raise IOError("fail")

    monkeypatch.setattr(tf.keras.models, "load_model", raise_exc)
    assert Inference.init("dummy_path") is False
    assert Inference.model is None


def test_run_with_model(monkeypatch):
    dummy_output = tf.ones((1, TARGET_SIZE, TARGET_SIZE, 3))
    dummy_model = mock.Mock(return_value=dummy_output)
    Inference.model = dummy_model
    inp = make_dummy_input()
    out = Inference.run(inp)
    assert isinstance(out, PredictionInput)
    np.testing.assert_allclose(out.image.numpy(), dummy_output.numpy() * 255)
    assert out.id == inp.id
    assert out.params == inp.params


def test_run_without_model_and_init_fails(monkeypatch):
    monkeypatch.setattr(Inference, "init", lambda *a, **kw: False)
    inp = make_dummy_input()
    out = Inference.run(inp)
    assert isinstance(out, PredictionInput)
    assert np.allclose(out.image.numpy(), 0)
    assert out.id == inp.id
    assert out.params == inp.params


def test_run_without_model_and_init_succeeds(monkeypatch):
    dummy_output = tf.ones((1, TARGET_SIZE, TARGET_SIZE, 3))
    dummy_model = mock.Mock(return_value=dummy_output)

    def fake_init(*a, **kw):
        Inference.model = dummy_model
        return True

    monkeypatch.setattr(Inference, "init", fake_init)
    inp = make_dummy_input()
    out = Inference.run(inp)
    assert isinstance(out, PredictionInput)
    np.testing.assert_allclose(out.image.numpy(), dummy_output.numpy() * 255)
    assert out.id == inp.id
    assert out.params == inp.params
