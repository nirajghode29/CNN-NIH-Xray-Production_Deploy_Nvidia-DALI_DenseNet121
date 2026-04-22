from io import BytesIO
from PIL import Image
import torch
from torchvision import transforms

# Replace these with the EXACT values used during training/inference
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = (244, 244)

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.Grayscale(num_output_channels=3),  # keep only if used in your training pipeline
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

def read_image_from_bytes(contents: bytes):
    image = Image.open(BytesIO(contents)).convert("RGB")
    return image

def preprocess_image(contents: bytes):
    image = read_image_from_bytes(contents)
    tensor = transform(image).unsqueeze(0)
    return tensor