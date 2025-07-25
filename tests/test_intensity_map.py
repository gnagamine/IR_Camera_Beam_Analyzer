import sys
import types
from pathlib import Path

import pytest

@pytest.fixture(scope="module")
def intensity_map():
    # Ensure project root is in sys.path for module import
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    # Create dummy modules for dependencies not installed in the test environment
    dummy_mpl = types.ModuleType("matplotlib")
    dummy_mpl.use = lambda *args, **kwargs: None
    dummy_pyplot = types.ModuleType("matplotlib.pyplot")
    dummy_mpl.pyplot = dummy_pyplot

    sys.modules.setdefault("matplotlib", dummy_mpl)
    sys.modules.setdefault("matplotlib.pyplot", dummy_pyplot)
    sys.modules.setdefault("numpy", types.ModuleType("numpy"))
    sys.modules.setdefault("pandas", types.ModuleType("pandas"))

    from IntensityMapIRCamera import IntensityMap
    return IntensityMap.__new__(IntensityMap)

def test_numeric_strings(intensity_map):
    assert intensity_map._is_cell_convertible_or_empty("1.2")
    assert intensity_map._is_cell_convertible_or_empty("3,4")

def test_empty_strings(intensity_map):
    assert intensity_map._is_cell_convertible_or_empty("")
    assert intensity_map._is_cell_convertible_or_empty("   ")

def test_non_numeric_text(intensity_map):
    assert not intensity_map._is_cell_convertible_or_empty("abc")
