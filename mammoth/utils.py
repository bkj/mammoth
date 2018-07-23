import os
import gzip
import struct
import array
import numpy as np
import pickle

def load_data(path='data/mnist_data.pkl', normalize=False):
    with open(path, 'rb') as f:
        train_images, train_labels, test_images, test_labels = pickle.load(f, encoding='latin1')
        
    one_hot = lambda x, K : np.array(x[:,None] == np.arange(K)[None, :], dtype=int)
    partial_flatten = lambda x : np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
    train_images = partial_flatten(train_images) / 255.0
    test_images  = partial_flatten(test_images)  / 255.0
    train_labels = one_hot(train_labels, 10)
    test_labels = one_hot(test_labels, 10)
    N_data = train_images.shape[0]
    
    if normalize:
        train_mean = np.mean(train_images, axis=0)
        train_images = train_images - train_mean
        test_images = test_images - train_mean
    
    return train_images, train_labels, test_images, test_labels, N_data