import pytest
import numpy as np
import json
from ..processing.colorscale import ScaleColor, ColorScale, COLORSCALES


def test_scalecolor_rgb():
    c = ScaleColor(1, 2, 3, 0.5)
    assert c.rgb == (1, 2, 3)


def test_scalecolor_background():
    bg = ScaleColor.background()
    assert bg.r == 0 and bg.g == 0 and bg.b == 0 and bg.value == -1


def test_colorscale_colors_lab():
    colors = [
        ScaleColor(255, 0, 0, 1),
        ScaleColor(0, 255, 0, 2),
        ScaleColor(0, 0, 255, 3),
    ]
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
    assert (
        cs2.colors[0].r == 10
        and cs2.colors[0].g == 20
        and cs2.colors[0].b == 30
        and cs2.colors[0].value == 1.5
    )


def test_colorscale_from_cloud_and_load(monkeypatch):
    # Mock load to return a valid JSON string
    json_input = '[{"Color": [1, 2, 3], "Value": 4.5}]'
    monkeypatch.setattr(
        "server.processing.colorscale.ColorScale.load",
        staticmethod(lambda cs: json_input),
    )
    cs = ColorScale.from_cloud(COLORSCALES.DF)
    assert isinstance(cs, ColorScale)
    assert cs.colors[0].r == 1


def test_colorscale_from_cloud_none(monkeypatch):
    # Mock load to return None
    monkeypatch.setattr(
        "server.processing.colorscale.ColorScale.load", staticmethod(lambda cs: None)
    )
    cs = ColorScale.from_cloud(COLORSCALES.DF)
    assert cs is None


def test_colorscale_load_success(monkeypatch):
    monkeypatch.setattr(
        "server.processing.colorscale.GCSManager.load",
        staticmethod(lambda path, bucket_name=None: "data"),
    )
    result = ColorScale.load(COLORSCALES.DF)
    assert result == "data"


def test_colorscale_load_exception(monkeypatch):
    def raise_exc(path, bucket_name=None):
        raise Exception("fail")

    monkeypatch.setattr(
        "server.processing.colorscale.GCSManager.load", staticmethod(raise_exc)
    )
    result = ColorScale.load(COLORSCALES.DF)
    assert result is None


def test_colorscales_enum():
    assert COLORSCALES.DF.value == "colorscale_df.json"
    assert COLORSCALES.DA.value == "colorscale_da.json"


def test_value_to_color_single():
    colors = [ScaleColor(0, 0, 0, 0.0), ScaleColor(255, 255, 255, 10.0)]
    cs = ColorScale(colors=colors)
    # Closest to 0.1 is index 0, closest to 9.9 is index 1
    out0 = cs.value_to_color(np.array([0.1]))
    out1 = cs.value_to_color(np.array([9.9]))
    assert out0[0].value == 0.0
    assert out1[0].value == 10.0


def test_value_to_color_multiple():
    colors = [
        ScaleColor(0, 0, 0, 0.0),
        ScaleColor(255, 255, 255, 10.0),
        ScaleColor(128, 128, 128, 5.0),
    ]
    cs = ColorScale(colors=colors)
    values = np.array([0.1, 4.9, 10.0])
    out = cs.value_to_color(values)
    # Should match closest color for each value
    assert [c.value for c in out] == [0.0, 5.0, 10.0]


def test_value_to_color_shape_and_type():
    colors = [ScaleColor(0, 0, 0, 0.0), ScaleColor(255, 255, 255, 10.0)]
    cs = ColorScale(colors=colors)
    values = np.array([[0.1, 9.9], [10.0, 0.0]])
    out = cs.value_to_color(values.flatten())
    assert isinstance(out, np.ndarray)
    assert all(isinstance(c, ScaleColor) for c in out)


def test_value_to_color_dtype_and_shape():
    colors = [ScaleColor(0, 0, 0, 0.0), ScaleColor(255, 255, 255, 10.0)]
    cs = ColorScale(colors=colors)
    # 1D input
    values_1d = np.array([0.1, 9.9])
    out_1d = cs.value_to_color(values_1d)
    assert isinstance(out_1d, np.ndarray)
    assert out_1d.dtype == object
    assert all(isinstance(c, ScaleColor) for c in out_1d)
    assert out_1d.shape == (2,)
    # 2D input
    values_2d = np.array([[0.1, 9.9], [10.0, 0.0]])
    out_2d = cs.value_to_color(values_2d.flatten())
    assert isinstance(out_2d, np.ndarray)
    assert out_2d.dtype == object
    assert all(isinstance(c, ScaleColor) for c in out_2d)
    assert out_2d.shape == (4,)
