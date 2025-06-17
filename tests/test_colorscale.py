import pytest
import numpy as np
from ..processing.colorscale import ScaleColor, ColorScale, COLORSCALES

def test_scalecolor_rgb():
    c = ScaleColor(1, 2, 3, 0.5)
    assert c.rgb == (1, 2, 3)

def test_scalecolor_background():
    bg = ScaleColor.background()
    assert bg.r == 0 and bg.g == 0 and bg.b == 0 and bg.value == -1

def test_colorscale_colors_lab():
    colors = [ScaleColor(255, 0, 0, 1), ScaleColor(0, 255, 0, 2), ScaleColor(0, 0, 255, 3)]
    cs = ColorScale(colors=colors)
    lab = cs.colors_lab
    assert isinstance(lab, np.ndarray)
    assert lab.shape == (3, 3)

def test_colorscale_to_json_and_from_json():
    colors = [ScaleColor(10, 20, 30, 1.5)]
    cs = ColorScale(colors=colors)
    json_str = cs.to_json()
    # Simulate expected JSON structure for from_json
    json_input = '[{"Color": [10, 20, 30], "Value": 1.5}]'
    cs2 = ColorScale.from_json(json_input)
    assert isinstance(cs2, ColorScale)
    assert cs2.colors[-1] == ScaleColor.background()
    assert cs2.colors[0].r == 10 and cs2.colors[0].g == 20 and cs2.colors[0].b == 30 and cs2.colors[0].value == 1.5

def test_colorscale_from_cloud_and_load(monkeypatch):
    # Mock load to return a valid JSON string
    json_input = '[{"Color": [1, 2, 3], "Value": 4.5}]'
    monkeypatch.setattr("server.processing.colorscale.ColorScale.load", staticmethod(lambda cs: json_input))
    cs = ColorScale.from_cloud(COLORSCALES.DF)
    assert isinstance(cs, ColorScale)
    assert cs.colors[0].r == 1

def test_colorscale_from_cloud_none(monkeypatch):
    # Mock load to return None
    monkeypatch.setattr("server.processing.colorscale.ColorScale.load", staticmethod(lambda cs: None))
    cs = ColorScale.from_cloud(COLORSCALES.DF)
    assert cs is None

def test_colorscale_load_success(monkeypatch):
    monkeypatch.setattr("server.processing.colorscale.GCSManager.load", staticmethod(lambda path, bucket_name=None: "data"))
    result = ColorScale.load(COLORSCALES.DF)
    assert result == "data"

def test_colorscale_load_exception(monkeypatch):
    def raise_exc(path, bucket_name=None):
        raise Exception("fail")
    monkeypatch.setattr("server.processing.colorscale.GCSManager.load", staticmethod(raise_exc))
    result = ColorScale.load(COLORSCALES.DF)
    assert result is None

def test_colorscales_enum():
    assert COLORSCALES.DF.value == "colorscale_df.json"
    assert COLORSCALES.DA.value == "colorscale_da.json"