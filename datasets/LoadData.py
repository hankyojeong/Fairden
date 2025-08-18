import pickle
import numpy as np
import pandas as pd


__datasets = ['toy', 'adult', 'bank(2)', 'bank(3)', 'diabetes']

def dataset_names():
    return __datasets


def L2_normalize(X):
    feanorm = np.maximum(1e-14, np.sum(X**2,axis=1))
    X_out = X / (feanorm[:,None]**0.5)
    return X_out


def load_data(name='adult', l2_normalize=False):

    if name not in __datasets:
        raise KeyError("Dataset not implemented: ", name)
    else:
        data_path = f'datasets/{name}/'
        inputs = np.load(data_path + 'inputs.npy')
        labels = np.load(data_path + 'labels.npy')
        sensitives = np.load(data_path + 'sensitives.npy')

        inputs_mean = np.mean(inputs, axis=0)
        inputs_std = np.std(inputs, axis=0) + 1e-10
        inputs = (inputs - inputs_mean) / inputs_std
        
        d = inputs.shape[1]
        n_color = np.unique(sensitives).shape[0]

    if l2_normalize:
        inputs = L2_normalize(inputs)

    return inputs, labels, sensitives, d, n_color