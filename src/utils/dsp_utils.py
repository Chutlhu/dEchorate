import numpy as np
import scipy.signal as sg

def normalize(x):
    return x/np.max(np.abs(x))

def center(x):
    return x - np.mean(x)

def envelope(x):
    return np.abs(sg.hilbert(x))


def make_toepliz_as_in_mulan(v, L):
    D = v.shape[0]
    T = np.zeros([D-L+1, L], dtype=np.complex64)
    R, _ = T.shape
    for r in range(R):
        T[r, :] = v[r:r+L][::-1]
    return T
