import numpy as np
import scipy.signal as sg

def normalize(x):
    return x/np.max(np.abs(x))

def center(x):
    return x - np.mean(x)

def envelope(x):
    return np.abs(sg.hilbert(x))
