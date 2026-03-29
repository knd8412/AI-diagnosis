import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "image_processing"))
import unittest
import numpy as np
from utils import validate_xray, format_output

class TestValidateXray(unittest.TestCase):

    def test_rejects_uniform_black(self):
        """Rejects a completely black image (too uniform)."""
        img = np.zeros((224, 224), dtype=np.uint8)
        with self.assertRaises(ValueError) as ctx:
            validate_xray(img)
        self.assertIn("uniform", str(ctx.exception).lower())

    def test_rejects_uniform_white(self):
        """Rejects a completely white image (too bright / too uniform)."""
        img = np.full((224, 224), 255, dtype=np.uint8)
        with self.assertRaises(ValueError) as ctx:
            validate_xray(img)
        self.assertTrue(
            "uniform" in str(ctx.exception).lower() or
            "bright" in str(ctx.exception).lower()
        )

    def test_rejects_uniform_rgb(self):
        """Rejects a flat RGB image (too uniform)."""
        img = np.full((224, 224, 3), 128, dtype=np.uint8)
        with self.assertRaises(ValueError) as ctx:
            validate_xray(img)
        self.assertIn("uniform", str(ctx.exception).lower())

    def test_rejects_too_bright_rgb(self):
        """Rejects an RGB image with very high brightness."""
        img = np.full((224, 224, 3), 240, dtype=np.uint8)
        with self.assertRaises(ValueError) as ctx:
            validate_xray(img)
        self.assertTrue(
            "bright" in str(ctx.exception).lower() or
            "uniform" in str(ctx.exception).lower()
        )

    def test_rejects_1d(self):
        img = np.zeros((224,))
        with self.assertRaises(ValueError):
            validate_xray(img)

    def test_rejects_too_many_channels(self):
        img = np.zeros((224, 224, 5))
        with self.assertRaises(ValueError):
            validate_xray(img)

class TestFormatOutput(unittest.TestCase):

    def test_filters_below_threshold(self):
        probs = {"Pneumonia": 0.8, "Effusion": 0.3, "Atelectasis": 0.6}
        result = format_output(probs, threshold=0.5)
        self.assertIn("Pneumonia", result)
        self.assertIn("Atelectasis", result)
        self.assertNotIn("Effusion", result)

    def test_empty_when_all_below_threshold(self):
        probs = {"Pneumonia": 0.1, "Effusion": 0.2}
        self.assertEqual(format_output(probs, threshold=0.5), {})
