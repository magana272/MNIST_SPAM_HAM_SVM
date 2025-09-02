import numpy as np
from typing import Dict

def get_indices(total_datasize, number_or_percent_or_all) -> None | Dict[str, np.ndarray[any, np.dtype[np.signedinteger[any]]]]:
    """
    Generate random indices for splitting data into training and validation sets.

    Args:
        total_datasize (int): Total number of samples in the dataset.
        number_or_percent (int | float): Number or fraction of samples to use for training.

    Returns:
        dict: Dictionary with 'training' and 'validation' indices as numpy arrays.
    """
    
    if(isinstance(number_or_percent_or_all, str) and number_or_percent_or_all!="ALL"):
        return None
    elif number_or_percent_or_all == "ALL":
        number_of_samples = total_datasize 
    elif (number_or_percent_or_all < 0) or (total_datasize == 0) or (number_or_percent_or_all > total_datasize):
        return None
    elif number_or_percent_or_all < 1:
        number_of_samples = int(total_datasize * number_or_percent_or_all)
    else:
        number_of_samples = number_or_percent_or_all

    indices = np.random.permutation(total_datasize)
    training = indices[:number_of_samples]
    validation = indices[number_of_samples:]
    return {"training": training, "validation": validation}


def preprocess_image(data:np.ndarray)-> np.ndarray:
    """
    Reshape image data to 2D array where each row is a flattened image.

    Args:
        data (np.ndarray): Image data of shape (n_samples, height, width).

    Returns:
        np.ndarray: Reshaped data of shape (n_samples, height*width).
    """
    return data.reshape(data.shape[0], 28*28)

def training_validation_split(number_or_percent: int | float, data: np.array, labels: np.array) -> tuple:
    """
    Split data and labels into training and validation sets.

    Args:
        number_or_percent (int | float): Number or fraction of samples to use for training.
        data (np.array): The input data array.
        labels (np.array): The corresponding labels array.

    Returns:
        tuple: (training_data, training_labels, validation_data, validation_labels)
    """

    total = data.shape[0]
    testindex_validationindices = get_indices(total, number_or_percent)
    training_indx = testindex_validationindices["training"]
    validation_indx = testindex_validationindices["validation"]
    return data[training_indx], labels[training_indx], data[validation_indx], labels[validation_indx]