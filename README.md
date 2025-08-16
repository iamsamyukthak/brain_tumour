
# Brain Tumour Detection using CNN / ResNet18

This project uses deep learning to classify brain MRI scans into:
- **Has Brain Tumour**
- **Doesn't have Brain Tumour**

## Goals
- Train a ResNet18 model on the Kaggle Brain MRI dataset.
- Evaluate accuracy, loss, and predictions.
- Save model weights for later inference.
- Provide a simple script (`predict.py`) to test on new MRI images.

## Dataset
We use the Kaggle dataset:  
[Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)

## How to Run
1. Train the model in Google Colab using `train.py`.
2. Save the trained model to `model/resnet18_brain_tumor.pt`.
3. Run `predict.py` with any MRI image:
   ```bash
   python predict.py --image your_image.jpg
