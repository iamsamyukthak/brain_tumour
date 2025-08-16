import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import argparse
import os
from PIL import Image

# For Grad-CAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def plot_metrics(history):
    """Plots the training and validation accuracy and loss."""
    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('model/training_metrics.png')
    print("✅ Training metrics plot saved to model/training_metrics.png")

def evaluate_model(model, test_loader, device):
    """Evaluates the model on the test set and prints a classification report."""
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    report = classification_report(y_true, y_pred, target_names=['No', 'Yes'])
    print("\nClassification Report:\n")
    print(report)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('model/confusion_matrix.png')
    print("✅ Confusion matrix saved to model/confusion_matrix.png")


def generate_grad_cam(model, target_layer, test_loader, device, num_images=4):
    """Generates and saves Grad-CAM heatmaps for a few test images."""
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=(device.type == 'cuda'))
    
    os.makedirs("grad_cam_examples", exist_ok=True)

    for i, (inputs, labels) in enumerate(test_loader):
        if i >= num_images:
            break
        
        input_tensor = inputs[0:1].to(device) # Get the first image
        rgb_img = inputs[0].permute(1, 2, 0).numpy()
        rgb_img = (rgb_img - np.min(rgb_img)) / (np.max(rgb_img) - np.min(rgb_img)) # Normalize to 0-1 for visualization

        targets = [ClassifierOutputTarget(labels[0].item())]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        # Save the image
        img = Image.fromarray(visualization)
        img.save(f"grad_cam_examples/example_{i+1}_label_{test_loader.dataset.classes[labels[0].item()]}.png")

    print(f"✅ Grad-CAM heatmaps saved to the 'grad_cam_examples/' directory.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the brain tumor classifier.')
    parser.add_argument('--model_path', type=str, default='model/resnet18_brain_tumor.pt', help='Path to the trained model.')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    
    # Load training history (assuming it's saved during training)
    try:
        history = torch.load('model/training_history.pt')
        plot_metrics(history)
    except FileNotFoundError:
        print("Could not find 'model/training_history.pt'. Skipping metrics plot.")

    # Prepare the test dataset
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_dataset = datasets.ImageFolder('data/Testing', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Run evaluation
    evaluate_model(model, test_loader, device)
    
    # Generate Grad-CAM
    target_layer = model.layer4[-1] # Target the last conv block of ResNet18
    generate_grad_cam(model, target_layer, test_loader, device)
