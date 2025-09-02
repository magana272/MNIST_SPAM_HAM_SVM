
import numpy as np
from abc import ABC, abstractmethod
from typing import NamedTuple
from typing_extensions import override

class DataSplit(NamedTuple):
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    
    
class KFoldIteratorInterface(ABC):
    @abstractmethod
    def next()->list[list[int]]:
        pass
    @abstractmethod
    def has_next()->bool:
        pass

class KfoldCrossValidation(KFoldIteratorInterface):
        def __init__(self, data:np.ndarray, labels:np.ndarray, K:int=5):
            super().__init__()
            
            if( K < 0 or K > len(data)):
                raise ValueError("K must be postive and be less then sample points")
            self.data = data
            self.labels = labels
            self.remaining = K
            self.splits_indxs : np.ndarray[np.ndarray[int]] | None = self.k_fold_cross_validation_operation(data, K)
            
        
        @override
        def has_next(self) -> bool:
            if self.remaining > 0:
                return True
            return False
        @override
        def next(self) -> DataSplit | None:
            if self.remaining > 0:
                validation_set:np.ndarray[int]|int = np.delete(self.splits_indxs, 0)
                training_set:np.ndarray[int]|int = self.splits_indxs.copy()
                res = DataSplit(x_train=self.data[training_set], y_train=self.labels[training_set], x_val=self.data[validation_set], y_val=self.labels[validation_set])
                self.splits_indxs  = np.concat((training_set, validation_set))
                self.remaining -=1
                return res
            else:
                return None
                
        
        def k_fold_cross_validation_operation(self, data:np.ndarray, K:int)-> np.ndarray[np.ndarray[int]] | None:

            """
            Generate random indices for splitting data into training and validation sets.

            Args:
                total_datasize (int): Total number of samples in the dataset.
                number_or_percent (int | float): Number or fraction of samples to use for training.

            Returns:
                dict: Dictionary with 'training' and 'validation' indices as numpy arrays.
            """

            if( K < 0 or K > len(data)):
                self.remaining = 0
                return None
            else:
                number_of_samples:int = data.shape[0]
                group_size:int  = int(number_of_samples/K)
                indices: list[int] = np.random.permutation(number_of_samples)
                groups: list[list[int]] = []
                left, right = 0, group_size
                while right < number_of_samples:
                    groups.append(indices[left:right])
                    left+=group_size
                    right+=group_size
                if(left < number_of_samples):
                    groups.append(indices[left:])
                return np.array(groups[0])
        
    