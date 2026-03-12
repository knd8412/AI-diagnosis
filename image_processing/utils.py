import logging
from skimage import exposure

def validate_xray(img):
    if img.ndim not in (2, 3):
        raise ValueError("Not a valid image: expected 2D or 3D array")
    if img.ndim == 3 and img.shape[-1] > 4:
        raise ValueError("Not grayscale: too many channels")

    if exposure.is_low_contrast(img): logging.warning("Low contrast image")

def format_output(probs, threshold=0.5):
    return {k: float(v) for k,v in probs.items() if v > threshold}
