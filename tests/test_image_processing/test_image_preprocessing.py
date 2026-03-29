import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "image_processing"))
import unittest
import numpy as np
import torch
from preprocess import preprocess, load_image, normalize

class TestNormalize(unittest.TestCase):

    def test_uint8_range(self):
        img = np.full((224, 224), 128, dtype=np.uint8)
        result = normalize(img)
        # 128/255 * 2048 - 1024 ≈ 4.0
        self.assertAlmostEqual(float(result.mean()), 4.0, delta=1.0)

    def test_float_unit_range(self):
        img = np.full((224, 224), 0.5, dtype=np.float32)
        result = normalize(img)
        self.assertAlmostEqual(float(result.mean()), 0.0, delta=1.0)

    def test_16bit_range(self):
        img = np.full((224, 224), 32767, dtype=np.uint16)
        result = normalize(img)
        self.assertTrue(-1024 <= result.mean() <= 1024)

class TestPreprocess(unittest.TestCase):

    def test_output_shape(self):
        img = np.random.randint(0, 255, (300, 300), dtype=np.uint8)
        tensor = preprocess(img)
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, torch.Size([1, 1, 224, 224]))

    def test_rgba_converted_to_grayscale(self):
        img = np.random.randint(0, 255, (300, 300, 4), dtype=np.uint8)
        tensor = preprocess(img)
        self.assertEqual(tensor.shape, torch.Size([1, 1, 224, 224]))

    def test_rgb_converted_to_grayscale(self):
        img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        tensor = preprocess(img)
        self.assertEqual(tensor.shape, torch.Size([1, 1, 224, 224]))
