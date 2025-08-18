*Brain Tumor Classification with PyTorch and ResNet18*
This project provides a deep learning solution for classifying brain MRI scans. It utilizes a pre-trained ResNet18 model to determine whether an MRI image indicates the presence of a brain tumor. Model interpretability is enhanced with Grad-CAM, which generates heatmaps to highlight the areas in an image that most influence the model's prediction.

*Key Features*
Accurate Classification: Distinguishes between MRIs with and without brain tumors.
Transfer Learning: Built upon a ResNet18 model pre-trained on ImageNet for robust feature extraction.
Model Interpretability: Implements Grad-CAM to visualize the model's decision-making process.
Complete Workflow: Includes scripts for training, evaluation, and prediction on new images.
Dataset
This model is trained on the Brain MRI Images for Brain Tumor Detection dataset from Kaggle.

*Project Goals*
Train a robust ResNet18 model on the Kaggle Brain MRI dataset.
Evaluate model performance using accuracy, loss, and a confusion matrix.
Save the model weights for easy inference and deployment.
Visualize model predictions using Grad-CAM to ensure reliability.

You can download the dataset here : https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection
