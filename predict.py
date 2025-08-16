import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models
import argparse

# Parse argument
parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, required=True, help="Path to MRI image")
args = parser.parse_args()

# Load model
model = models.resnet18(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model.load_state_dict(torch.load("model/resnet18_brain_tumor.pt", map_location=torch.device('cpu')))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load image
image = Image.open(args.image).convert("RGB")
image = transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    if predicted.item() == 0:
        print("Doesn't have brain tumour")
    else:
        print("Has brain tumour")
