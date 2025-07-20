import os
import sys
import pytest

# Ensure scripts package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.eval import is_equiv as eval_is_equiv
from scripts.openrouter_eval import is_equiv as router_is_equiv

@pytest.mark.parametrize("func", [eval_is_equiv, router_is_equiv])
def test_invalid_input_returns_false(func):
    assert func(1, "1") is False
    assert func("1", 1) is False
