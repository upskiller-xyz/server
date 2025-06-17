import numpy as np
import pytest
from unittest import mock

from ..processing.colorconverter import ColorConverter

class DummyColor:
    def __init__(self, rgb):
        self.rgb = rgb

class DummyColorScale:
    @property
    def colors_lab(self):
        # 3 colors in Lab space
        return [(10, 20, 30), (40, 50, 60), (70, 80, 90)]
    @property
    def colors(self):
        return [DummyColor((1,2,3)), DummyColor((4,5,6)), DummyColor((7,8,9))]

def test_init():
    cc = ColorConverter(colorscale=DummyColorScale())
    kmeans = cc.init()
    assert hasattr(kmeans, "fit")

def test_to_lab(monkeypatch):
    cc = ColorConverter(colorscale=DummyColorScale())
    img = np.ones((2,2,3), dtype=np.uint8) * 255
    monkeypatch.setattr("cv2.cvtColor", lambda arr, code: arr + 1)
    out = cc.to_lab(img)
    assert np.all(out == img + 1)

def test_make(monkeypatch):
    cc = ColorConverter(colorscale=DummyColorScale())
    img = np.zeros((2,2,3), dtype=np.uint8)
    monkeypatch.setattr(cc, "to_lab", lambda x: np.zeros((2,2,3)))
    out = cc.make(img)
    assert out.shape == (2,2,3)

def test_cluster(monkeypatch):
    cc = ColorConverter(colorscale=DummyColorScale())
    img = np.ones((2,2,3), dtype=np.uint8)
    dummy_kmeans = mock.Mock()
    dummy_kmeans.fit.return_value = "fitted"
    cc.clustering = dummy_kmeans
    monkeypatch.setattr(cc, "to_lab", lambda x: img)
    res = cc.cluster(img)
    assert res == "fitted"

def test_label(monkeypatch):
    cc = ColorConverter(colorscale=DummyColorScale())
    img = np.ones((2,2,3), dtype=np.uint8)
    dummy_kmeans = mock.Mock()
    dummy_kmeans.fit.return_value = dummy_kmeans
    dummy_kmeans.labels_ = np.arange(4)
    cc.clustering = dummy_kmeans
    monkeypatch.setattr(cc, "cluster", lambda x: dummy_kmeans)
    labels = cc.label(img)
    assert labels.shape == (2,2)
    assert np.all(labels.flatten() == np.arange(4))

def test__make(monkeypatch):
    cc = ColorConverter(colorscale=DummyColorScale())
    img = np.ones((2,2,3), dtype=np.uint8)
    dummy_kmeans = mock.Mock()
    dummy_kmeans.fit.return_value = dummy_kmeans
    dummy_kmeans.labels_ = np.zeros(4, dtype=int)
    cc.clustering = dummy_kmeans
    monkeypatch.setattr(cc, "cluster", lambda x: dummy_kmeans)
    out = cc._make(img)
    assert out.shape == (2,2,3)