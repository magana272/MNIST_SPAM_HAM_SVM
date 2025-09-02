import numpy as np

def classification_accuracy(actual_labels : np.ndarray, predicted_labels: np.ndarray):
    
    """
    Calculate the classification accuracy as the proportion of correct predictions.

    Args:
        actual_labels (np.ndarray): True labels.
        predicted_labels (np.ndarray): Predicted labels.

    Returns:
        float: Classification accuracy.
    """
    return np.mean(actual_labels == predicted_labels)

