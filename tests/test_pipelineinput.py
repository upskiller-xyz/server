import pytest
import tensorflow as tf
from ..processing.pipelineinput import PipelineInput, GetOneDfPipelineInput, GetDfPipelineInput

def test_pipelineinput_default_and_build():
    p = PipelineInput.default()
    assert isinstance(p, PipelineInput)
    assert p.value == 0
    p2 = PipelineInput.build(42)
    assert isinstance(p2, PipelineInput)
    assert p2.value == 42

def test_getonedfpipelineinput_build(monkeypatch):
    dummy_img = tf.ones((8, 8, 3))
    monkeypatch.setattr("server.processing.image_manager.ImageManager.get_image", lambda b: dummy_img)
    inp = GetOneDfPipelineInput.build("bytes")
    assert isinstance(inp, GetOneDfPipelineInput)
    assert tf.reduce_all(inp.value == dummy_img)

def test_getdfpipelineinput_build(monkeypatch):
    dummy_img = tf.ones((8, 8, 3))
    monkeypatch.setattr("server.processing.image_manager.ImageManager.get_image", lambda b: dummy_img)
    result = GetDfPipelineInput.build(["a", "b"])
    # Note: build returns a list, not an instance!
    assert isinstance(result, list)
    assert all(tf.reduce_all(img == dummy_img) for img in result)