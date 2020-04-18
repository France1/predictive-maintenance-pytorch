import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader
import torch


class PumpDataset(Dataset):
    def __init__(self, X, y):
        """Pytorch dataset for pump condition based maintenance data
        
        Arguments:
            X : array (samples, channel, sequence)
                measurements from sensors
            y : array (samples, labels)
                labels of system conditions
                
        Returns:
            sample : dict[torch.tensor]
                dictionary containing for a single data sample sensor measurements and labels         
        """
        self.X = X
        self.y = y
                
    def __len__(self):
        
        return len(self.X)
    
    def __getitem__(self, idx):
        sequence = self.X[idx]
        label = self.y[idx]    
        sample = {'sequence': torch.tensor(sequence, dtype=torch.float),
                  'label': torch.tensor(label, dtype=torch.long)}
        
        return sample
    

def make_dataset(data, channels, labels):
    """Prepare training and testing data set for deep learning 
    
    Dataset is split with 10% data for testing and shuffled with stratified split. Sensor data are combined into a 2D data, then shuffled-split, then reshaped into 3D arrays
    
    Inputs:
        data : dict[array]
            dictionary containing sensor and labels input data
        channels: list[srt]
            list of channels (sensor labels) to include in the data set
        channels: list[srt]
            list of condition labels to include in the data set

    Outputs:
        X_train: array (n_train, channels, sequence)
            training measurements from sensors (channels)
        y_train : array (samples, labels)
            training labels of system conditions (labels)
        X_test: array (n_test, channels, sequence)
            test measurements from sensors (channels)
        y_test : array (samples, labels)
            test labels of system conditions (labels)   
    """
    # dictionary -> array (sequence*samples, channels)
    X = np.concatenate([np.expand_dims(data[channel],axis=2) for channel in channels], axis=2)
    y = data['labels'][:,labels]
    
    sss = StratifiedShuffleSplit(test_size=0.1, random_state=40)
    
    for train_index,test_index in sss.split(X,y):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test  = X[test_index]
        y_test  = y[test_index]
        
    # array (sequence*samples, channels) -> (samples, channels, sequence)
    X_train = np.reshape(X_train, (len(X_train), len(channels), -1))
    X_test = np.reshape(X_test, (len(X_test), len(channels), -1))
    
    return X_train, y_train, X_test, y_test


def make_loaders(data, channels, labels, batch_size=4):
    """Prepare pytorch dataloaders  
    
    Inputs:
        data : dict[array]
            dictionary containing sensor and labels input data
        channels: list[srt]
            list of channels (sensor labels) to include in the data set
        channels: list[srt]
            list of condition labels to include in the data set
        batch_size: int
            number of data samples in each batch
            
    Outputs:
        train_loader: object
            pytorch training set dataloader 
        test_loader: object
            pytorch test set dataloader   
    """
    
    X_train, y_train, X_test, y_test = make_dataset(data, channels, labels)
    train_dataset = PumpDataset(X_train, y_train) 
    test_dataset = PumpDataset(X_test, y_test)
    # use drop_last option to exclude the last incomplete batch which causes errors
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)
    
    return train_loader, test_loader
