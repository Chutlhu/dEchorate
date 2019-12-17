import numpy as np
import scipy as sp
from scipy.linalg import toeplitz

import matplotlib.pyplot as plt

def normalize_and_zero_mean(x):
    x -= np.mean(x)
    x /= np.max(np.abs(x))
    return x


def awgn(x, snr_db):

    if snr_db > 100:
        print('SNR more the 100 dB, considered as noiseless')
        return x

    L = max(x.shape)
    snr_lin = 10**(snr_db/10)   # SNR to linear scale
    Ex = np.sum(np.abs(x)**2)/L # Energy of the signal
    N = Ex/snr_lin              # find the noise variance
    n = np.sqrt(N)              # standard deviation for AWGN noise
    return x + np.random.normal(0, n, L)


def vincent_deconvolution(y, x):
    '''
    Deconvolution with Vincent's method: X = Y/H
    '''
    if not y.shape == x.shape:
        raise ValueError('y and h should be of the same size')
    h = np.real(np.fft.ifft(np.fft.fft(y)/np.fft.fft(x)))
    return h


def koldovsky_deconvolution(x, y, Lh, delay=0, lam=0):
    '''
    Deconvolution with Koldowsky's method: using time-domain LS

    Least-square time-domain estimator of relative impulse response (relative
    transfer function) between xL and xR.

    input:
    y : recorded signal
    x : reference signal
    Lh : length of estimated IR
    delay : global delay of the estimated relative impulse response
            due to causality
    lam : regularization parameter

    output:
    g : estimated relative impulse response- xR relative to xL
    G : estimated relative transfer function (RTF)
    '''

    if not y.shape == x.shape:
        raise ValueError('y and h should be of the same size')

    λ = lam
    Lh -= 1

    # ensure channel causality
    x = np.concatenate([np.zeros(int(delay)), x])

    ##### Σxx ###############
    # 1. auto covariance of x
    Σxx = np.zeros([Lh+1,1])
    r = np.correlate(x, x, 'full') # equal to matlab's xcorr
     # select the part for delay=0 to delay=Lh
    r = r[len(x)-Lh-1:len(x)]

    # Tikhonov regularization
    # vector which only its first value differs from 0
    Σxx = (λ * np.max(r)*np.eye(Lh+1, 1)).squeeze()

    # Take only the first half of r, and reverse its order,
    # s.t. r(N+1)-->R(0)
    # Then, add the regularization "reg*max(r)" to R(0)
    Σxx += r[::-1]

    ##### Σyx ###########################
    # 2. crosscorrelation between x and y
    p = np.correlate(y, x, 'full') # equal to matlab's xcorr
    # Take only the second half of p
    p = p[len(x)-1:len(x)+Lh]
    Σyx = p

    ### Toepz(Σxx)\Σyx ###################
    # 3.a fast computation of the division
    h = block_levinson(Σyx[:, None], Σxx[:, None])

    return h


def olivier_deconvolution(y, x):
    '''
    Deconvolution with Olivier's method: using cross-correlation
    '''
    return ValueError('Not working well')
    if not y.shape == x.shape:
        raise ValueError('y and h should be of the same size')
    return np.real(np.fft.ifft(np.conj(np.fft.fft(x[::-1]))*np.fft.fft(y)))


def wiener_deconvolution(y, x, pad):
    X = np.fft.fft(x, pad)
    Y = np.fft.fft(y, pad)
    Sxx = X * np.conj(X)
    Syx = Y * np.conj(X)
    H = Syx / Sxx
    return np.real(np.fft.ifft(H))


def envelope(x):
    return np.abs(sp.signal.hilbert(x))


def find_delay(x):
    return delay


def make_same_length(x, y, kind='max', pad=0):
    Nx = len(x)
    Ny = len(y)
    if kind == 'max':
        N = max(Nx, Ny) + pad
        xo = np.zeros(N)
        yo = np.zeros(N)
        xo[:Nx] = x
        yo[:Ny] = y
    elif kind == 'min':
        N = min(Nx, Ny)
        xo = np.zeros(N)
        yo = np.zeros(N)
        xo[:N] = x[:N]
        yo[:N] = y[:N]
    return xo, yo


def block_levinson(y, L):
    '''
    adapted from Zbynek algo
    '''

    zeros = lambda n: np.zeros([n, n])
    eye = lambda n: np.eye(n)

    Lr, Lc = L.shape
    d = Lc          # Block dimension
    N = Lr // d     # Number of Blocks

    # This is just to get the bottom block row B
    # from the left block column L
    B = np.reshape(L, [d, N, d])
    B = B.transpose([0, 2, 1])
    B = np.flip(B, axis=2)
    B = np.reshape(B, [d, N*d])

    f = np.linalg.inv(L[:d,:])  # "Forward" block vector
    b = f                        # "Backward" block vector
    x = f @ y[:d]                # Solution vector

    for n in range(2, N+1):
        ef = B[:, (N-n)*d:N*d] @ np.concatenate([f, zeros(d)], axis=0)
        eb = L[:n*d, :].T      @ np.concatenate([zeros(d), b], axis=0)
        ex = B[:, (N-n)*d:N*d] @ np.concatenate([x, zeros(d)], axis=0)

        A1 = np.concatenate([eye(d), eb], axis=1)
        A2 = np.concatenate([ef, eye(d)], axis=1)
        A = np.concatenate([A1, A2], axis=0)
        A = np.linalg.inv(A)

        fn1 = np.concatenate([f, zeros(d)], axis=0)
        fn2 = np.concatenate([zeros(d), b], axis=0)
        fn = np.concatenate([fn1, fn2], axis=1)
        fn = fn @ A[:, :d]

        bn1 = np.concatenate([f, zeros(d)], axis=0)
        bn2 = np.concatenate([zeros(d), b], axis=0)
        bn = np.concatenate([fn1, fn2], axis=1)
        bn = bn @ A[:, d:]

        f = fn.copy()
        b = bn.copy()
        x1 = np.concatenate([x, np.zeros([d, 1])], axis=0)
        x = x1 + b @ (y[(n-1)*d:n*d] - ex)

    return x.squeeze()


