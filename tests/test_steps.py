import pytest
import mock
import sys
sys.path.append("..")

from ..processing.steps.step import Step, PredictStep, ImageCombineStep, ImageAlignStep

class DummyPipelineInput:
    def __init__(self, value=None):
        self.value = value

@pytest.fixture
def dummy_input():
    return DummyPipelineInput(value="dummy_value")

def test_step_run_calls__run(monkeypatch, dummy_input):
    called = {}
    class DummyStep(Step):
        @classmethod
        def _run(cls, inp, config):
            called['yes'] = True
            return inp
    out = DummyStep.run(dummy_input)
    assert out is dummy_input
    assert called['yes']

def test_predict_step_run(monkeypatch, dummy_input):
    dummy_result = "predicted"
    monkeypatch.setattr("server.processing.steps.step.Inference.run", lambda v: dummy_result)
    dummy_input.value = "input_for_inference"
    out = PredictStep._run(dummy_input)
    assert out == dummy_result

def test_image_combine_step_run(dummy_input):
    # As implemented, just returns input
    out = ImageCombineStep._run(dummy_input)
    assert out is dummy_input

def test_image_align_step_run(monkeypatch, dummy_input):
    dummy_result = "aligned"
    monkeypatch.setattr("server.processing.steps.step.ImageTransformer.run", lambda inp: dummy_result)
    out = ImageAlignStep._run(dummy_input)
    assert out == dummy_result