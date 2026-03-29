import requests
from typing import Dict

# Configuration
IMAGE_PROCESSING_URL = "http://image-processing-service:8001"
REQUEST_TIMEOUT = 30

# Reusable HTTP Session
_session = requests.Session()


def analyze_xray_image(image_path: str) -> Dict:
    """
    Send X-ray to image processing service.

    Args:
        image_path: Path to X-ray image file

    Returns:
        dict: {
            "pathologies": {...},
            "confident_results": {...},
            "scan_file": "...",
            "result_id": "..."
        }

    Raises:
        RuntimeError: With specific message depending on failure type
    """
    try:
        with open(image_path, 'rb') as f:
            response = _session.post(
                f"{IMAGE_PROCESSING_URL}/analyze",
                files={'file': f},
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            return response.json()

    except FileNotFoundError:
        raise RuntimeError(f"X-ray file not found: {image_path}")
    except requests.exceptions.Timeout:
        raise RuntimeError(f"Image processing service timed out after {REQUEST_TIMEOUT}s")
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"Could not reach image processing service at {IMAGE_PROCESSING_URL}")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"Image processing returned {e.response.status_code}: {e.response.text}")
    except ValueError:
        raise RuntimeError("Image processing service returned invalid JSON")


def extract_pathology_symptoms(analysis_result: Dict) -> str:
    """
    Convert pathology results to a human-readable symptom description.

    Args:
        analysis_result: Response from /analyze endpoint

    Returns:
        str: Human-readable symptom description
    """
    if not analysis_result or not isinstance(analysis_result, dict):
        return "No analysis result available"

    confident = analysis_result.get("confident_results", {})

    if not confident:
        return "No significant pathologies detected in X-ray"

    symptoms = [
        f"{pathology} (confidence: {round(confidence * 100)}%)"
        for pathology, confidence in confident.items()
    ]

    return "X-ray findings: " + ", ".join(symptoms)