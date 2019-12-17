import pytest

import numpy as np
from scipy.linalg import toeplitz

from src.dsp_utils import block_levinson

def test_block_levinson():

    x = 10*np.random.random([20,5]) - 5
    t = 5*np.random.random(20) - 2.5
    T = toeplitz(t)

    y = T @ x

    x_est = block_levinson(y, T)

    assert np.allclose(x, x_est)