import pytest
from unittest import mock

from ..processing.pipeline import Pipeline, GetOneDfPipeline, GetDfPipeline


class DummyInput:
    def __init__(self, value):
        self.value = value


@pytest.fixture
def dummy_input():
    return DummyInput(value=[1, 2, 3])


def test_pipeline_run_calls_steps(monkeypatch):

    called = []

    class DummyStep:
        @classmethod
        def run(cls, inp):
            called.append(inp)
            return inp

    ss = [DummyStep, DummyStep]

    class DummyPipeline(Pipeline):
        steps = ss

    inp = DummyInput(value=42)
    out = DummyPipeline.run(inp)
    assert out is inp
    assert called == [inp, inp]


def test_get_one_df_pipeline_steps(monkeypatch):
    # Patch PredictStep and ImageAlignStep
    monkeypatch.setattr(
        "server.processing.steps.step.PredictStep.run",
        lambda inp, config=None: DummyInput(value="pred"),
    )
    monkeypatch.setattr(
        "server.processing.steps.step.ImageAlignStep.run",
        lambda inp, config=None: DummyInput(value="aligned"),
    )
    inp = DummyInput(value="img")
    out = GetOneDfPipeline.run(inp)
    assert isinstance(out, DummyInput)
    assert out.value == "aligned"


def test_get_df_pipeline(monkeypatch):
    # Patch MultiThreader and steps
    monkeypatch.setattr(
        "server.processing.pipeline.MultiThreader.run",
        lambda func, inps: [DummyInput(value="pred1"), DummyInput(value="pred2")],
    )
    monkeypatch.setattr(
        "server.processing.pipeline.inpt.GetOneDfPipelineInput", lambda v: v
    )
    monkeypatch.setattr(
        "server.processing.pipeline.inpt.GetDfPipelineInput",
        lambda vals: DummyInput(value=vals),
    )
    monkeypatch.setattr(
        "server.processing.pipeline.GetOneDfPipeline.run",
        lambda inp, config=None: DummyInput(value="aligned"),
    )
    monkeypatch.setattr(
        "server.processing.steps.step.ImageCombineStep.run",
        lambda inp, config=None: DummyInput(value="combined"),
    )
    inp = DummyInput(value=["img1", "img2"])
    out = GetDfPipeline.run(inp)
    assert isinstance(out, DummyInput)
    assert out.value == "combined"
