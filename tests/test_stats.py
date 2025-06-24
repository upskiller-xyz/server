import numpy as np
import pytest
from ..processing.stats import Stat, AvgStat, Gt1Stat, StatsPack, STATS


class DummyImageTransformer:
    @staticmethod
    def get_mask(matrix):
        # Mask: True for values > 0, False otherwise
        return matrix > 0


def test_stat_out_and_build(monkeypatch):
    # Patch ImageTransformer.get_mask to select all nonzero
    monkeypatch.setattr(
        "server.processing.image_transformer.ImageTransformer.get_mask",
        DummyImageTransformer.get_mask,
    )
    arr = np.array([[0, 2], [3, 0]], dtype=float)
    stat = Stat.build(arr)
    assert isinstance(stat, Stat)
    assert isinstance(stat.out, dict)
    assert "Base" in stat.out


def test_avgstat(monkeypatch):
    monkeypatch.setattr(
        "server.processing.image_transformer.ImageTransformer.get_mask",
        DummyImageTransformer.get_mask,
    )
    arr = np.array([[0, 2], [3, 0]], dtype=float)
    avg = AvgStat.build(arr)
    assert isinstance(avg, AvgStat)
    # Only nonzero values: 2, 3
    assert np.isclose(avg.value, 2.5)
    assert avg.out == {"average_value": avg.value}


def test_gt1stat(monkeypatch):
    # For Gt1Stat, get_mask returns the mask itself, so _area is the mask
    monkeypatch.setattr(
        "server.processing.image_transformer.ImageTransformer.get_mask",
        DummyImageTransformer.get_mask,
    )
    arr = np.array([[0, 2], [3, 0]], dtype=float)
    gt1 = Gt1Stat.build(arr)
    assert isinstance(gt1, Gt1Stat)
    # _area = arr > 0: [[False, True], [True, False]]
    # np.sum(_area > 1.0) = 0 (since _area is bool)
    # np.sum(_area) = 2
    assert gt1.value == 0.0
    assert gt1.out == {"ratio_gt1": gt1.value}


def test_statspack(monkeypatch):
    monkeypatch.setattr(
        "server.processing.image_transformer.ImageTransformer.get_mask",
        DummyImageTransformer.get_mask,
    )
    arr = np.array([[0, 2], [3, 0]], dtype=float)
    pack = StatsPack.build(arr)
    assert isinstance(pack, StatsPack)
    out = pack.out
    assert isinstance(out, dict)
    # Should have EXTERNAL_KEYS.METRICS.value as key
    from server.processing.external import EXTERNAL_KEYS

    assert EXTERNAL_KEYS.METRICS.value in out
    # Each stat out should be a dict with the stat name as key
    metrics = out[EXTERNAL_KEYS.METRICS.value]
    assert isinstance(metrics, set) or isinstance(metrics, dict)
