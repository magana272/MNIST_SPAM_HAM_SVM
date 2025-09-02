import numpy as np


def load_all_data_sets() -> dict[dict[str, np.array]]:
    """
    Load all available datasets (toy, mnist, spam) from the data directory.

    Returns:
        dict: A dictionary where each key is a dataset name and each value is a loaded npz object containing the dataset.
    """
    all_data_sets = {}
    for data_name in ["toy", "mnist", "spam"]:
        data = np.load(f"./data/{data_name}-data.npz")
        print("\nloaded %s data!" % data_name)
        all_data_sets[data_name] = data
    return all_data_sets
def load_mnist() -> dict[dict[str, np.array]]:
    """
    Load the MNIST dataset from the data directory.

    Returns:
        dict: A loaded npz object containing the MNIST dataset.
    """
    data_name = "mnist"
    return np.load(f"./data/{data_name}-data.npz")
def load_spam() -> dict[dict[str, np.array]]:
    """
    Load the MNIST dataset from the data directory.

    Returns:
        dict: A loaded npz object containing the MNIST dataset.
    """
    data_name = "spam"
    return np.load(f"./data/{data_name}-data.npz")
