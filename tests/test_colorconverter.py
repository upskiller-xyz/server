import numpy as np
import pytest
from unittest import mock

from ..processing.colorconverter import ColorConverter


class DummyColor:
    def __init__(self, rgb, lab, value):
        self.rgb = rgb
        self.lab = lab
        self.value = value


class DummyColorScale:
    @property
    def colors_lab(self):
        # 3 colors in Lab space
        return [(10, 20, 30), (40, 50, 60), (70, 80, 90)]

    @property
    def colors(self):
        return [
            DummyColor((1, 2, 3), (10, 20, 30), 0.1),
            DummyColor((4, 5, 6), (40, 50, 60), 0.2),
            DummyColor((7, 8, 9), (70, 80, 90), 0.3),
        ]

    def value_to_color(self, labels):
        # Return DummyColor objects for each label
        arr = np.array([self.colors[l] for l in labels.flatten()]).reshape(labels.shape)
        return arr


def test_make_and_solve(monkeypatch):
    # Patch colorscale
    monkeypatch.setattr(ColorConverter, "colorscale", DummyColorScale())
    # Patch to_lab to identity for simplicity
    monkeypatch.setattr(ColorConverter, "to_lab", lambda img: img)
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    # Should return colors_lab[labels]
    out = ColorConverter.make(img)
    assert out.shape == (2, 2, 3)


def test_solve_returns_labels(monkeypatch):
    monkeypatch.setattr(ColorConverter, "colorscale", DummyColorScale())
    monkeypatch.setattr(
        ColorConverter, "to_lab", lambda img: np.zeros((2, 2, 3), dtype=np.uint8)
    )
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    labels = ColorConverter.solve(img)
    assert labels.shape == (2, 2)
    assert np.issubdtype(labels.dtype, np.integer)


def test_get_values(monkeypatch):
    monkeypatch.setattr(ColorConverter, "colorscale", DummyColorScale())
    monkeypatch.setattr(
        ColorConverter, "to_lab", lambda img: np.zeros((2, 2, 3), dtype=np.uint8)
    )
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    values = ColorConverter.get_values(img)
    assert values.shape == (2, 2)
    assert np.all(np.isfinite(values))


def test_label(monkeypatch):
    monkeypatch.setattr(ColorConverter, "colorscale", DummyColorScale())
    monkeypatch.setattr(
        ColorConverter, "to_lab", lambda img: np.zeros((2, 2, 3), dtype=np.uint8)
    )
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    labels = ColorConverter.label(img)
    assert labels.shape == (2, 2)


def test_values_to_image(monkeypatch):
    monkeypatch.setattr(ColorConverter, "colorscale", DummyColorScale())
    labels = np.zeros((2, 2), dtype=int)
    img = ColorConverter.values_to_image(labels)
    assert img.shape == (2, 2, 3)
    assert img.dtype == np.uint8
