#!/usr/bin/env python

"""
    utils.py
"""

import pickle
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(path='data/mnist_data.pkl', test_frac=0.25):
    X_train, y_train, X_test, y_test = pickle.load(open(path, 'rb'), encoding='latin1')
    
    partial_flatten = lambda x : np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
    X_train = partial_flatten(X_train) / 255.0
    X_test  = partial_flatten(X_test)  / 255.0
    
    train_mean = X_train.mean()
    train_std  = X_train.std()
    
    X_train = (X_train - train_mean) / train_std
    X_test  = (X_test - train_mean) / train_std
    
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=test_frac)
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test