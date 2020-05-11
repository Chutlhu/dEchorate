import numpy as np
import scipy as sp

from src.externals.trilaterate import trilaterate

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


def trilateration2(anchors, distances):
    return trilaterate(['a',['a']])


def trilateration(anchors, distances, init=None):
    # function [estimatedLocation, totalError] = trilaterate_beck(anchors, distances)
    # %------------------------------------------------------------------------------
    # % Trilaterates the location of a point from distances to a fixed set of
    # % anchors. Uses algorithm described by Beck et al.
    # %
    # % INPUT : anchors           ... anchor locations - if we have M anchors in D
    # %                               dimensions, a is an M by D matrix
    # %         distances         ... distances between anchors and the point of
    # %                               interest
    # %
    # % OUTPUT: estimatedLocation ... estimated location (D by 1)
    # %         totalError        ... sum of absolute distance errors from the
    # %                               estimated point to the anchors
    # %------------------------------------------------------------------------------
    # print(anchors.shape) # M x D
    # print(distances.shape)  # M x 1

    assert len(distances.shape) == 1
    assert anchors.shape[1] in [1, 2, 3]
    assert anchors.shape[0] == distances.shape[0]

    # d = size(anchors, 2);
    d = anchors.shape[1]
    # m = length(distances);
    m = len(distances)

    # A = [-2 * anchors, ones(m, 1)];
    A = np.concatenate([-2 * anchors, np.ones([m, 1])], axis = 1)
    assert A.shape == (m, d+1)
    # b = distances.^2 - sum(anchors.^2, 2);
    b = distances**2 - np.sum(anchors**2, 1)
    assert len(b) == m
    b = b.reshape([m, 1])

    # D = [eye(d), zeros(d, 1); zeros(1, d), 0];
    D = np.concatenate([np.eye(d), np.zeros([d, 1])], axis=1)
    D = np.concatenate([D, np.zeros([1, d+1])], axis=0)
    assert D.shape == (d+1, d+1)

    # f = [zeros(d, 1); -0.5];
    f = np.zeros([d+1, 1])
    f[-1] = -0.5

    # y   = @(lambda) (A'*A + lambda * D) \ (A'*b - lambda * f);
    def y(x):
        # phi = @(lambda) y(lambda)' * D * y(lambda) + 2 * f' * y(lambda);
        num = (A.T @ b - x * f)
        rden = np.linalg.pinv(A.T@A + x * D)
        a = rden @ num
        assert a.shape == (d + 1, 1)
        return a

    def phi(x):
        p = (y(x).T @ D @ y(x) + 2 * f.T @ y(x)).squeeze()
        return p

    eigDAA = sp.linalg.eigvals(D, A.T @ A)
    lambda1 = eigDAA[-1]

    a1 = -1 / lambda1
    a2 = 1000

    epsAbs = 1e-6
    epsStep = 1e-6

    # warning off;
    # while (a2 - a1 >= epsStep || ( abs( phi(a1) ) >= epsAbs && abs( phi(a2) )  >= epsAbs ) )
    while (a2 - a1) >= epsStep \
            or (np.abs(phi(a1)) >= epsAbs and np.abs(phi(a2) >= epsAbs)):
        c = (a1 + a2) / 2
        if (phi(c) == 0):
            break
        elif (phi(a1) * phi(c) < 0):
            a2 = c
        else:
            a1 = c


    estimatedLocation = np.real(y(c)[:d, :])

    # totalError = sum(abs(sqrt(sum(bsxfun(@minus, anchors', estimatedLocation).^2)) - distances(:)'))
    estimatedDistances = np.sqrt(np.sum((anchors.T - estimatedLocation)**2, 0))
    totalError = np.sum(np.abs(estimatedDistances - distances))
    return estimatedLocation, totalError
