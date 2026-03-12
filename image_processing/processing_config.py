import torch

MODEL_WEIGHTS = "densenet121-res224-all" 
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# PATHOLOGIES = [] 
