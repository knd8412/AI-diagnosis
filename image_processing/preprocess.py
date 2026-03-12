import torchxrayvision as xrv
import torchvision.transforms as T
import torch
import numpy as np
from skimage import io

def load_image(file_bytes_or_path):
    if isinstance(file_bytes_or_path, str):
        return io.imread(file_bytes_or_path)
    else:
        import io as _io
        return io.imread(_io.BytesIO(file_bytes_or_path))


def normalize(img):
    img = img.astype(np.float32)

    if img.max() <= 1.0:
        # Already in [0,1] float range
        img = img * 2048 - 1024
    elif img.max() <= 255:
        # Standard uint8 PNG/JPG
        img = img / 255.0 * 2048 - 1024
    elif img.max() <= 65535:
        # 16-bit image
        img = img / 65535.0 * 2048 - 1024
    else:
        # Assume DICOM-like Hounsfield, clip to safe range
        img = np.clip(img, -1024, 1024)
    return img


def preprocess(img):
    img = normalize(img)
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[:, :, :3]
        img = img.mean(2)
    img = img[None, ...]
    transform = T.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])
    img = transform(img).astype(np.float32)
    return torch.from_numpy(img).unsqueeze(0)

