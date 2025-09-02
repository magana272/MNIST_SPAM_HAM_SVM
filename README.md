# SVM MNIST & HAM/Spam Classification

This project implements Support Vector Machine (SVM) models for classification tasks on the MNIST dataset and a HAM/Spam text dataset. It includes modules for data loading, preprocessing, training, evaluation, and hyperparameter tuning.

## Project Structure

- `lib/` - Core Python modules for SVM training, evaluation, preprocessing, and utilities.
  - `accuracy.py` - Functions for computing classification accuracy.
  - `k_fold_cross_validation.py` - K-fold cross-validation implementation.
  - `load.py` - Data loading utilities for MNIST and HAM/Spam datasets.
  - `preprocess.py` - Data preprocessing functions.
  - `train.py` - SVM training routines.
- `data/` - Datasets and related files.
  - `mnist-data.npz` - MNIST dataset in NumPy format.
  - `spam-data.npz` - Spam dataset in NumPy format.
  - `toy-data.npz` - Toy dataset for testing.
  - `ham/` - HAM text files.
  - `spam/` - Spam text files.
  - `test/` - Test data files.
- `SVM_Evaluation_MNIST_Spam.ipynb` - Jupyter notebook for running experiments and visualizations.
- `requirements.txt` - Python dependencies for the project.
- `README.md` - Project documentation.

## Getting Started

### Prerequisites
- Python 3.8+
- Recommended: Create a virtual environment

### Installation
1. Clone the repository or download the project files.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
- Run the Jupyter notebook `SVM_Evaluation_MNIST_Spam.ipynb` for step-by-step experiments and results.
- Use the scripts in `lib/` for custom training and evaluation workflows.

## Features
- SVM training and evaluation on MNIST and HAM/Spam datasets
- Data preprocessing and feature extraction
- K-fold cross-validation for robust model assessment
- Hyperparameter tuning and regularization parameter optimization
- Accuracy computation and reporting

## Data
- MNIST: Handwritten digit images
- HAM/Spam: Text files for email classification

