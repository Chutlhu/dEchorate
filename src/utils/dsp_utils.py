import numpy as np

def normalize(x):
    return x/np.max(np.abs(x))

def center(x):
    return x - np.mean(x)