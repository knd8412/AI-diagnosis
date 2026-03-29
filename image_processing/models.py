import torch
import torchxrayvision as xrv
import numpy as np
import skimage.transform
from processing_config import MODEL_WEIGHTS, DEVICE

# =============================================================
# XRayModel wrapper
# =============================================================
# This class is a wrapper around torchxrayvision DenseNet.
# =============================================================

class XRayModel:
    def __init__(self):
        try:
            self.model = xrv.models.DenseNet(weights=MODEL_WEIGHTS).to(DEVICE)
            self.model.eval()
            self.pathologies = self.model.pathologies
        except Exception as e:
            raise RuntimeError(f"Failed to initialize XRayModel: {e}")

    def predict(self, img_tensor):
        # Function note:
        # - accepts a single-image tensor [1,1,224,224] from preprocess
        # - throws meaningful runtime errors for tracing data flow in the endpoint
        # - keeps return type simple dict[pathology -> score]
        if img_tensor is None:
            raise ValueError("img_tensor is None")

        try:
            with torch.no_grad():
                preds = self.model(img_tensor.to(DEVICE))
                probs = torch.sigmoid(preds).cpu().numpy()[0]
            return {str(k): float(v) for k, v in zip(self.pathologies, probs)}
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")

    def features(self, img_tensor):
        # Function note:
        # - returns spatial features used for CAM visualization
        # - if the depth doesn't match expected value host code can catch RuntimeError

        if img_tensor is None:
            raise ValueError("img_tensor is None")

        try:
            with torch.no_grad():
                feats = self.model.features(img_tensor.to(DEVICE))
            return feats.cpu().numpy()[0]  # (1024, 7, 7)
        except Exception as e:
            raise RuntimeError(f"Feature extraction failed: {e}")

    def generate_cam(self, spatial_feats, class_idx):
        weights = self.model.classifier.weight[class_idx].detach().cpu().numpy()  # (1024,)
        cam = np.einsum('c,chw->hw', weights, spatial_feats)  # (H,W)
        cam = skimage.transform.resize(cam, (224, 224), preserve_range=True)
        return cam

    def top_cams(self, spatial_feats, preds, top_k=10):
        
        top_pathologies = sorted(preds.items(), key=lambda x: x[1], reverse=True)[:top_k]
        cams = {}
        
        for pathology, score in top_pathologies:
            class_idx = self.pathologies.index(pathology)
            cam = self.generate_cam(spatial_feats, class_idx)
            cams[pathology] = cam.tolist()
        
        return cams

