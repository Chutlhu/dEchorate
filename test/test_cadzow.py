import pytest
import numpy as np

from src.utils.dsp_utils import *

def test_reconstruct_toeplitz():
    N = 10
    K = 4
    x = np.random.randn(N)

    Tx = make_toepliz_as_in_mulan(x, K)
    xr = reconstruct_toeplitz(Tx)
    assert np.allclose(xr, x)

def test_reshape_into_smaller_toeplitz():
    N = 10
    P = 4
    K = 2

    x = np.random.randn(N)

    Ap = make_toepliz_as_in_mulan(x, P)
    assert Ap.shape == (N-P+1, P)
    Ak = make_toepliz_as_in_mulan(x, K)
    assert Ak.shape == (N-K+1, K)

    print(Ap.shape)
    print(Ak.shape)

    Ap2k = reshape_toeplitz(Ap, K)
    assert Ap2k.shape == Ak.shape
    print(Ap2k.shape)

    assert np.allclose(Ap2k, Ak)

def test_toeplitz_as_mulan():
    N = 5
    K = 2
    x = np.random.randn(N) + 1j*np.random.randn(N)
    Th1 = make_toepliz_as_in_mulan(x, K)
    Th2 = make_toepliz_as_in_mulan2(x, K)
    assert Th1.shape == (N-K+1, K)
    assert Th1.shape == Th2.shape
    assert np.allclose(Th1, Th2)

def test_frobenius_weights():
    N = 20
    L = 8
    x = np.random.randn(N) + 1j*np.random.randn(N)
    Tx = make_toepliz_as_in_mulan(x, L)

    W = build_frobenius_weights(Tx)

    assert np.allclose(W[0,:], np.arange(1, W.shape[1]+1)[::-1])
    assert W.shape == Tx.shape
