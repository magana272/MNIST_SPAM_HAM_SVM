
from multiprocessing.pool import Pool
import numpy as np
from sklearn import svm
import time

from .load import load_mnist
from .preprocess import preprocess_image, training_validation_split
from .accuracy import classification_accuracy




def work(hp, training_data, training_labels, validation_data, validation_labels)-> dict[str, any]:
    """
    Docstring for work
    
    :param hp: regualaarization h
    :param training_data: 
    :param training_labels: 
    :param validation_data: 
    :param validation_labels: 
    :return: 
    :rtype: dict[str, Any]
    """
    trained_model = train_regularization_parameter(training_data, training_labels, hp)
    training_res = trained_model.predict(training_data)
    training_error =  classification_accuracy(training_res, training_labels)
    validation_res = trained_model.predict(validation_data)
    validation_error =  classification_accuracy(validation_res, validation_labels)
    print(hp, "Done \n")
    return {"c":str(hp), "trained_model":trained_model,"training_error": training_error , "validation_error":validation_error}

def compare_single_vs_multi(training_data, training_labels, validation_data, validation_labels, hyperparameters):
    def work_single(hp):
        trained_model = train_regularization_parameter(training_data, training_labels, hp)
        training_error = classification_accuracy(trained_model.predict(training_data), training_labels)
        validation_error = classification_accuracy(trained_model.predict(validation_data), validation_labels)
        return {"c": hp, "training_error": training_error, "validation_error": validation_error}

    start = time.time()
    results_single = [work_single(c) for c in hyperparameters]
    single_time = time.time() - start

    data_for_pool = [(c, training_data, training_labels, validation_data, validation_labels) for c in hyperparameters]

    start = time.time()
    with Pool() as p:
        results_multi = p.starmap(work, data_for_pool)
    multi_time = time.time() - start

    return single_time, multi_time, results_single, results_multi

def train(training_data: np.ndarray, training_labels: np.ndarray) -> svm.LinearSVC:
    """
    Train a LinearSVC model on the provided MNIST training data and labels.

    Args:
        training_data (np.ndarray): Training data samples.
        training_labels (np.ndarray): Corresponding labels for training data.

    Returns:
        svm.LinearSVC: Trained LinearSVC model.
    """
    untrained_model = svm.LinearSVC()
    return untrained_model.fit(training_data, training_labels)

def train_regularization_parameter(training_data: np.ndarray, training_labels: np.ndarray, c_hp: float) -> svm.LinearSVC:
    """
    Train a LinearSVC model on the provided MNIST training data and labels.

    Args:
        training_data (np.ndarray): Training data samples.
        training_labels (np.ndarray): Corresponding labels for training data.

    Returns:
        svm.LinearSVC: Trained LinearSVC model.
    """
    untrained_model = svm.LinearSVC(C = c_hp)
    return untrained_model.fit(training_data, training_labels)

def mnist_pipeline_hyperparamter_tuning(hyperparameters, training_size) -> dict[str, dict[str, float]]:
    """
     mnist_pipeline_HyperparamterTuning
    
    :param hyperparameters: 
    :return: 
    :rtype: dict[str, dict[str, float]]
    """
    # Load data
    mnist = load_mnist()
    mnist_training_data = mnist["training_data"]
    mnist_training_labels = mnist["training_labels"]
    # Clean the data 
    preprocess_mnist_training_data = preprocess_image(mnist_training_data)
    # Split data into Test and validations sets
    training_data, training_labels, validation_data, validation_labels = training_validation_split(training_size, preprocess_mnist_training_data, mnist_training_labels)
    # initiliaze the pool
    data = [(c, training_data, training_labels, validation_data, validation_labels) for c in hyperparameters]

    with Pool() as p:
        results = p.starmap(work, data) 
    
    error_rate_diff_hp: dict[str, dict[str, float]] = {}
    for res in results:
        error_rate_diff_hp[res["c"]] = {"training_error":res["training_error"], "validation_error":res["validation_error"], "trained_model": res["trained_model"]}
    return error_rate_diff_hp

