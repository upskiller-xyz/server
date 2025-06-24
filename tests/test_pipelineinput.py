import pytest
import base64
import numpy as np
import tensorflow as tf
from ..processing.pipelineinput import (
    PipelineInput,
    GetOneDfPipelineInput,
    GetDfPipelineInput,
)


def test_pipelineinput_default_and_build():
    p = PipelineInput.default()
    assert isinstance(p, PipelineInput)
    assert p.value == 0
    p2 = PipelineInput.build(42)
    assert isinstance(p2, PipelineInput)
    assert p2.value == 42


def test_getonedfpipelineinput_build(monkeypatch):
    dummy_img = np.ones((8, 8, 3))
    monkeypatch.setattr(
        "server.processing.image_manager.ImageManager.get_image", lambda b: dummy_img
    )
    bs = base64.b64encode(dummy_img.tobytes()).decode("utf-8")
    inp = GetOneDfPipelineInput.build(bs)
    assert isinstance(inp, GetOneDfPipelineInput)
    assert tf.reduce_all(inp.value.np_image == dummy_img)


def test_getdfpipelineinput_build(monkeypatch):
    dummy_img = np.ones((8, 8, 3))
    monkeypatch.setattr(
        "server.processing.image_manager.ImageManager.get_image", lambda b: dummy_img
    )
    result = GetDfPipelineInput.build([dummy_img])
    assert isinstance(result, GetDfPipelineInput)
    assert all(tf.reduce_all(img == dummy_img) for img in result.value)
