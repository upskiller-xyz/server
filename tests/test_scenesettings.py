from ..processing.image_transformer import SceneSettings
from ..processing.utils import TARGET_SIZE

def test_get_scale_x():
    # axis=0 should use width
    expected = TARGET_SIZE / SceneSettings.width
    assert SceneSettings.get_scale(0) == expected

def test_get_scale_y():
    # axis=1 should use height
    expected = TARGET_SIZE / SceneSettings.height
    assert SceneSettings.get_scale(1) == expected

def test_get_scale_default_axis():
    # Default axis=0
    expected = TARGET_SIZE / SceneSettings.width
    assert SceneSettings.get_scale() == expected