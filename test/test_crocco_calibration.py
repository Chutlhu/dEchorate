import pytest

import numpy as np
from src.calibration_and_mds import *


def test_edm():
    N = 5
    M = 6

    X = np.random.random([3, N])
    A = np.random.random([3, M])

    D = edm(X, A)
    for i in range(N):
        for j in range(M):
            assert np.allclose(D[i,j], np.linalg.norm(X[:,i] - A[:,j]))


def test_gradd_calibration_noiseless():
    room_size = [5, 6, 7]
    N = 5
    M = 6

    R = np.array(room_size)[:, None]
    Xtrue = R*np.random.random([3, N])
    Atrue = R*np.random.random([3, M])

    Dobs = edm(Xtrue, Atrue)
    Xest, Aest = nlls_mds(Dobs, Xtrue, Atrue)

    assert np.allclose(Xest, Xtrue)
    assert np.allclose(Aest, Atrue)


def test_gradd_calibration_noisy():
    room_size = [4, 5, 7]
    N = 30
    M = 4
    SNR = 0.10

    R = np.array(room_size)[:, None]
    Xtrue = R*np.random.random([3, N])
    Atrue = R*np.random.random([3, M])

    Xobs = Xtrue + SNR*np.random.randn(*Xtrue.shape)
    Aobs = Atrue + SNR*np.random.randn(*Atrue.shape)

    Dobs = edm(Xtrue, Atrue)
    Xest, Aest = nlls_mds(Dobs, Xobs, Aobs)

    print(np.abs(Xest - Xtrue))
    assert np.all(np.abs(Xest - Xtrue)**2 < 0.005)

    print(np.abs(Aest - Atrue))
    assert np.all(np.abs(Aest - Atrue)**2 < 0.005)


def test_gradd_calibration_noisy():
    room_size = [4, 5, 7]
    N = 10
    M = 5
    SNR = 0.10

    R = np.array(room_size)[:, None]
    Xtrue = R*np.random.random([3, N])
    Atrue = R*np.random.random([3, M])

    Xobs = Xtrue + SNR*np.random.randn(*Xtrue.shape)
    Aobs = Atrue + SNR*np.random.randn(*Atrue.shape)

    Dobs = edm(Xtrue, Atrue)
    Xest, Aest = crcc_mds1(Dobs, Xobs, Aobs)

    print(np.abs(Xest - Xtrue))
    assert np.all(np.abs(Xest - Xtrue)**2 < 0.005)

    print(np.abs(Aest - Atrue))
    assert np.all(np.abs(Aest - Atrue)**2 < 0.005)


def test_crcc_calibration_noiseless():
    room_size = [5, 6, 7]
    I = 5
    J = 6

    R = np.array(room_size)[:, None]
    Xtrue = R*np.random.random([3, I])
    Atrue = R*np.random.random([3, J])

    Dobs = edm(Xtrue, Atrue)
    Xest, Aest = crcc_mds1(Dobs, Xtrue, Atrue)

    assert np.allclose(Xest, Xtrue)
    assert np.allclose(Aest, Atrue)
