import numpy as np


def edm(X, Y):
    '''
    Return Euclidean Distance Matrix
    s.t. D[i,j] = np.linalg.norm(X[:, i] - Y[:, i])
    '''
    Dx, N = X.shape
    Dy, M = Y.shape
    assert (Dx == Dy == 3) or (Dx == Dy == 2)

    # norm_X2 = sum(X. ^ 2);
    norm_X2 = np.sum(X ** 2, axis=0)[None, :]
    # norm_Y2 = sum(Y. ^ 2);
    norm_Y2 = np.sum(Y ** 2, axis=0)[None, :]
    # D = bsxfun(@plus, norm_X2', norm_Y2) - 2*X'*Y;
    D = norm_X2.T + norm_Y2 - 2 * X.T @ Y
    return D**0.5
