import logging
import numpy as np
from skimage import exposure

# =============================================================
# IMAGE PROCESSING UTILS (STUDENT PROJECT)
# =============================================================
# Usage:
#   - validate_xray() throws ValueError for invalid cases
#   - intended for experimentation and not complete clinical validation
#   - feel free to augment with size/type checks and better thresholds
# =============================================================

def validate_xray(img):
    """
    Validate that image is likely a medical X-ray

    Function note:
      - raises ValueError for obvious fails (dimensionality, intensity statistics)
      - warns for low contrast but does not reject (tunable)
      - this is a best-effort image quality filter, not medical diagnosis
    """
    if img.ndim not in (2, 3):
        raise ValueError("Not a valid image: expected 2D or 3D array")
    
    if img.ndim == 3 and img.shape[-1] > 4:
        raise ValueError("Not grayscale: too many channels")

    if img.ndim == 3:
        img_gray = np.mean(img, axis=2)
    else:
        img_gray = img

    # Relaxed: was 0.1, very uniform images (solid color) still caught
    std_dev = np.std(img_gray)
    if std_dev < 0.02:
        raise ValueError("Image appears too uniform to be an X-ray")

    mean_intensity = np.mean(img_gray)
    if mean_intensity > 0.95:
        raise ValueError("Image is too bright to be a typical X-ray")
    
    if exposure.is_low_contrast(img):
        logging.warning("Low contrast image — may not be a valid X-ray")

def format_output(probs, threshold=0.5):
    """
    Filter predictions by confidence threshold.
    
    Args:
        probs (dict): Dictionary of condition: probability pairs
        threshold (float): Minimum probability to include in output
    
    Returns:
        dict: Filtered predictions above threshold
    """
    return {condition: prob for condition, prob in probs.items() if prob >= threshold}