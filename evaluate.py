'''
Loads your trained ResNet18 model.
Shows training history (if available).
Evaluates on test set with classification report + confusion matrix.
Displays Grad-CAM heatmaps for a few MRI test images.'''

%%writefile evaluate.py
import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
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
    # Instead of saving, we display the plot
    plt.show()
    print("✅ Training metrics plot displayed above.")

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
    # Instead of saving, we display the plot
    plt.show()
    print("✅ Confusion matrix displayed above.")


def generate_grad_cam(model, target_layer, test_loader, device, num_images=4):
    """Generates and displays Grad-CAM heatmaps for a few test images."""
    cam = GradCAM(model=model, target_layers=[target_layer])

    # Setup for plotting in a grid
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    if num_images == 1:
        axes = [axes]
    fig.suptitle('Grad-CAM Heatmaps', fontsize=16)

    images_shown = 0
    data_iter = iter(test_loader)

    while images_shown < num_images:
        try:
            inputs, labels = next(data_iter)
            input_img, label = inputs[0], labels[0]
        except StopIteration:
            break

        input_tensor = input_img.unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            _, pred_idx = torch.max(output, 1)
            predicted_class = test_loader.dataset.classes[pred_idx.item()]
            actual_class = test_loader.dataset.classes[label.item()]

        rgb_img = input_img.permute(1, 2, 0).numpy()
        rgb_img = (rgb_img - np.min(rgb_img)) / (np.max(rgb_img) - np.min(rgb_img))

        targets = [ClassifierOutputTarget(label.item())]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # Displaying the image in its subplot
        ax = axes[images_shown]
        ax.imshow(visualization)
        ax.set_title(f"Actual: {actual_class}\nPredicted: {predicted_class}")
        ax.axis('off')
        images_shown += 1

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Display the entire figure with all subplots
    plt.show()
    print(f"✅ Grad-CAM heatmaps displayed above.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the brain tumor classifier.')
    parser.add_argument('--model_path', type=str, default='model/resnet18_brain_tumor.pt', help='Path to the trained model.')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)

    try:
        history = torch.load('model/training_history.pt')
        plot_metrics(history)
    except FileNotFoundError:
        print("Could not find 'model/training_history.pt'. Skipping metrics plot.")

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_dataset = datasets.ImageFolder('data/Testing', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    evaluate_model(model, test_loader, device)

    target_layer = model.layer4[-1]
    generate_grad_cam(model, target_layer, test_loader, device)
