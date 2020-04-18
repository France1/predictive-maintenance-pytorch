import numpy as np
import os
import pickle
import copy
from natsort import natsorted
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
import pandas as pd

'''
Preparation of dataset according to information contained in:

N. Helwig, et al., "Condition monitoring of a complex hydraulic system using multivariate statistics," 2015 IEEE International Instrumentation and Measurement Technology Conference (I2MTC) Proceedings, Pisa, 2015, pp. 210-215.

For a summarized description refer to description.txt and documentation.txt
'''

T_CYCLE = 60 # duration of each sequence in the dataset [sec] - not affecting the analysis

def load_data(data_dir, files): 
    """Load dataset consisting of a series of .txt files - one for each measuring sensor.

    Inputs:
        data_dir : str
            path of directory of the dataset
        files : list[str]
            list of file labels to load

    Outputs:
        data : dict[array]
            dictionary mapping each sensor label to corresponding data array
    """
    data = {}
    for file in files:
        data[file] = np.loadtxt(os.path.join(data_dir, file + '.txt'))
    
    return data

def interpolate_data(data, n_i):
    """Interpolate sensor data into a single array with common sampling rate
    
    Different sensors have different sampling rate, interpolation in a unique time sequence is necessary to analyse different sensors data simultaneously 

    Inputs:
        data : dict[array]
            dictionary mapping each file to corresponding np.array of data
        n_i : int
            number of resampling data points

    Outputs:
        data_int : dict[array]
            dictionary mapping each file sensor to interpolated data array 
    """
    x_i = np.linspace(0,T_CYCLE,n_i)
    data_int = {}
    channels = [c for c in data.keys() if c!='profile']
    for channel in [key for key in channels]:
        if data[channel].shape[1] != n_i:
            x = np.linspace(0,T_CYCLE,len(data[channel][0]))
            Y = data[channel]
            f = interp1d(x, Y, kind='linear')
            Y_i = f(x_i)
            data_int[channel] = Y_i
        else:
            data_int[channel] = data[channel]
            
    return data_int

def scale_data(data, n_i): 
    """Standardize measurements by removing the mean and scaling to unit variance
    
    Reshape dataset dictionary into a 2d array, apply scaling, then reshape back into dictionary
    
    Inputs:
        data_int : dict[array]
            dictionary mapping each file sensor to interpolated data array
        n_i : int
            number of resampling data points - needed for reshaping

    Outputs:
        data_scaled : dict[array]
            dictionary mapping each file sensor to scaled data array 
    """
    array = np.concatenate([v.reshape(-1,1) for v in data.values()], axis=1)
    array = StandardScaler().fit_transform(array)
    data_scaled = {}
    for i,k in enumerate(data.keys()):
        data_scaled[k] = array[:,i].reshape(-1, n_i)
        
    return data_scaled

def make_labels(data, data_scaled):
    """Add condition monitoring labels to scaled dataset
    
    For each system conditions (4 in total) convert conditions values into 0,1,..,N and add to scaled data set
    
    Inputs:
        data : dict[array]
            dictionary of raw data - containing system conditions
        data_scaled : dict[array]
            dictionary mapping each file sensor to scaled data array

    Outputs:
        data_scaled : dict[array]
            dictionary mapping each file sensor to scaled data array and conditions labels 
    """
    
    labels = []
    for i in range(4):
        label_i = data['profile'][:,i].astype(int)
        for j,val in enumerate(np.sort(np.unique(label_i))):
            label_i[label_i==val] = j
        labels.append(label_i.reshape(-1,1))
    labels = np.concatenate(labels, axis=1)
    data_scaled['labels'] = labels
    
    return data_scaled

def import_data(data_dir, n_i=6000):
    """Load and transform data into a dataset for machine learning analysis
    
    An optimal list of files correlated less than 95% is defined according to the analysis in Data_preparation.ipynb. 
        
    Inputs:
        data_dir : str
            path of directory of the dataset
        n_i : int
            number of resampling data points

    Outputs:
        dataset : dict[array]
            dictionary containing transformed sensor data and labels 
    """
    
    files = ['CP','FS1','PS1','PS2','PS3','PS4','PS5','SE', 'VS1','profile']
    data = load_data(data_dir, files)
    data_val = interpolate_data(data,n_i)
    data_val = scale_data(data_val,n_i)
    dataset = make_labels(data, data_val)
    
    return dataset  
    
