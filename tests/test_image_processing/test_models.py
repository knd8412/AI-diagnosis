import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "image_processing"))
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import torch

class TestXRayModel(unittest.TestCase):

    def _make_mock_densenet(self, MockDenseNet):
        mock_net = MagicMock()
        mock_net.to.return_value = mock_net
        mock_net.pathologies = ["Pneumonia", "Effusion", "Atelectasis"]
        mock_net.return_value = torch.tensor([[2.0, -1.0, 0.5]])
        mock_net.features.return_value = torch.zeros(1, 1024, 7, 7)
        mock_net.classifier.weight = torch.nn.Parameter(torch.rand(3, 1024))
        MockDenseNet.return_value = mock_net
        return mock_net

    @patch("models.xrv.models.DenseNet")
    def test_predict_returns_dict(self, MockDenseNet):
        self._make_mock_densenet(MockDenseNet)
        from models import XRayModel
        model = XRayModel()
        tensor = torch.zeros(1, 1, 224, 224)
        result = model.predict(tensor)
        self.assertIsInstance(result, dict)
        self.assertIn("Pneumonia", result)

    @patch("models.xrv.models.DenseNet")
    def test_predict_probabilities_in_range(self, MockDenseNet):
        self._make_mock_densenet(MockDenseNet)
        from models import XRayModel
        model = XRayModel()
        result = model.predict(torch.zeros(1, 1, 224, 224))
        for v in result.values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    @patch("models.xrv.models.DenseNet")
    def test_features_shape(self, MockDenseNet):
        self._make_mock_densenet(MockDenseNet)
        from models import XRayModel
        model = XRayModel()
        feats = model.features(torch.zeros(1, 1, 224, 224))
        self.assertEqual(feats.shape, (1024, 7, 7))

    @patch("models.xrv.models.DenseNet")
    def test_top_cams_keys_match_top_k(self, MockDenseNet):
        self._make_mock_densenet(MockDenseNet)
        from models import XRayModel
        model = XRayModel()
        spatial_feats = np.zeros((1024, 7, 7))
        preds = {"Pneumonia": 0.9, "Effusion": 0.4, "Atelectasis": 0.6}
        cams = model.top_cams(spatial_feats, preds, top_k=2)
        self.assertEqual(len(cams), 2)
        self.assertIn("Pneumonia", cams)


