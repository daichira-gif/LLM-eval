
import unittest

from scripts import eval as eval_module
from scripts import openrouter_eval as oe

class IsEquivInvalidInputTests(unittest.TestCase):
    def test_eval_invalid_input(self):
        self.assertFalse(eval_module.is_equiv("a", 1))

    def test_openrouter_invalid_input(self):
        self.assertFalse(oe.is_equiv("a", 1))

if __name__ == "__main__":
    unittest.main()

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

