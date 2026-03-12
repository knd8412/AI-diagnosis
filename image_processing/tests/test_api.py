import requests
import json

URL = "http://localhost:8001/analyze"
IMAGE_PATH = "/Users/grigory/Downloads/sample_xray.png"

def test_service():
    print(f"Sending image to {URL}...")
    
    try:
        with open(IMAGE_PATH, "rb") as image_file:
            files = {"file": (IMAGE_PATH, image_file, "image/png")}
            
            response = requests.post(URL, files=files)
            
            if response.status_code == 200:
                print("Success! Response:")
                data = response.json()
                if "spatial_features" in data:
                    data["spatial_features"] = f"Vector of length {len(data['spatial_features'])}"
                    
                print(json.dumps(data, indent=2))
            else:
                print(f"Error {response.status_code}: {response.text}")
                
    except FileNotFoundError:
        print(f"Please provide a valid image at {IMAGE_PATH}")
    except requests.exceptions.ConnectionError:
        print(f"Could not connect to {URL}. Is the Docker container running?")

if __name__ == "__main__":
    test_service()
