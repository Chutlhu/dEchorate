import numpy as np
import scipy as sp
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


def make_toepliz_as_in_mulan2(v, L):
    D = len(v)
    r1 = v[:L][::-1]
    c1 = v[L-1:]
    return sp.linalg.toeplitz(c1, r1)

def reconstruct_toeplitz(Ta):
    # to reconstruct the toeplitz take the last column (-1 last element)
    # and the last row in reverse
    return np.concatenate([Ta[:-1, -1], Ta[-1, :][::-1]])

def reshape_toeplitz(Ta, L):
    a = reconstruct_toeplitz(Ta)
    return make_toepliz_as_in_mulan(a, L)
