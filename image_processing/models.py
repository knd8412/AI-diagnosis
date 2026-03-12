import torch
import torchxrayvision as xrv
import numpy as np
import skimage.transform
from processing_config import MODEL_WEIGHTS, DEVICE

class XRayModel:
    def __init__(self):
        self.model = xrv.models.DenseNet(weights=MODEL_WEIGHTS).to(DEVICE)
        self.model.eval()
        self.pathologies = self.model.pathologies

    def predict(self, img_tensor):
        with torch.no_grad():
            preds = self.model(img_tensor.to(DEVICE))
            probs = torch.sigmoid(preds).cpu().numpy()[0]
        return {str(k): float(v) for k, v in zip(self.pathologies, probs)}

    def features(self, img_tensor):
        with torch.no_grad():
            feats = self.model.features(img_tensor.to(DEVICE))
        return feats.cpu().numpy()[0]  # (1024, 7, 7)

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

