import torchxrayvision as xrv
import torchvision.transforms as T
import torch
import numpy as np
from skimage import io

# =============================================================
# IMAGE PROCESSING PREPROCESS
# =============================================================
# This module handles basic image loading and normalization.
# In a more complete pipeline, consider:
#   - strict type checks (np.ndarray, bytes, strings)
#   - image format assertions (grayscale / medical-specific)
# =============================================================

def load_image(file_bytes_or_path):
    # Function note:
    # - supports file path and raw bytes input
    # - raises ValueError for invalid images, which is caught by the endpoint
    # - this is where you could add explicit MIME-type guard logic
    try:
        if isinstance(file_bytes_or_path, str):
            return io.imread(file_bytes_or_path)
        else:
            import io as _io
            return io.imread(_io.BytesIO(file_bytes_or_path))
    except Exception as e:
        raise ValueError(f"Could not load image: {e}")


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
    # Function note:
    # - should receive a 2D/3D numpy array in HxW or HxWxC format
    # - this is where additional checks (e.g., whether DICOM, size) are reasonable
    # - output is a 1x1x224x224 tensor for the model, so classifier pipeline expects this
    if img is None:
        raise ValueError("Image data is None")

    img = normalize(img)

    if img.ndim not in (2, 3):
        raise ValueError("Unsupported image dimensions, expected 2D or 3D")

    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[:, :, :3]
        img = img.mean(2)

    img = img[None, ...]
    transform = T.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])

    try:
        img = transform(img).astype(np.float32)
        return torch.from_numpy(img).unsqueeze(0)
    except Exception as e:
        raise RuntimeError(f"Failed to preprocess image: {e}")

