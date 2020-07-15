import numpy as np
import scipy as sp
import scipy.signal as sg

import librosa as lr

def normalize(x):
    return x/np.max(np.abs(x))

def center(x):
    return x - np.mean(x)

def envelope(x):
    return np.abs(sg.hilbert(x))


def todB(x):
    return 10*np.log10(x)


def rake_filter(ak, tk, omegas):
    assert len(ak) == len(tk)
    assert len(ak.shape) == len(tk.shape) == 1

    H = np.exp(-1j*omegas[:, None] @ tk[None, :]) @ ak[:, None]
    assert H.shape == (len(omegas), 1)

    return H.squeeze()


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


def build_frobenius_weights(A):
    N, L = A.shape
    D = N + L - 1

    # matrix of weights for the weighted Frobenius norm
    r = np.arange(1, L+1)[::-1]
    c = np.concatenate([np.arange(1,L+1), L*np.ones(N-L)])[::-1]
    W = sp.linalg.toeplitz(c, r)
    return W

def enforce_toeplitz(A):
    N, P = A.shape
    z = np.zeros(N + P - 1, dtype=np.complex64)
    for i in range(z.shape[0]):
        z[i] = np.mean(np.diag(A, P - i - 1))

    return make_toepliz_as_in_mulan(z, P)


def resample(x, old_fs, new_fs):
    return lr.resample(x.T, old_fs, new_fs).T
