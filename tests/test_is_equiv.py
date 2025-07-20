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
