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


def trilateration(anchor_pos, distances):
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
    print(anchor_pos.shape)
    print(distances.shape)

    assert distances.shape[1] == 1
    assert anchor_pos.shape[0] in [1, 2, 3]
    assert anchor_pos.shape[1] == distances.shape[0]

    # matlab notation
    # distances  # D x 1
    anchors = anchor_pos.T  # M x D

    # d = size(anchors, 2);
    d = anchor_pos.shape[1]
    # m = length(distances);
    m = len(distance)

    # A = [-2 * anchors, ones(m, 1)];
    A = np.concatenate([-2 * anchor_pos, np.ones(m, 1)])
    # b = distances.^2 - sum(anchors.^2, 2);
    b = distance**2 - np.sum(anchors**2, 1)

    # D = [eye(d), zeros(d, 1); zeros(1, d), 0];
    D = np.concatenate([np.eye(d), np.zeros(d, 1)], axis=1)
    D = np.concatenate([D, np.zeros(1, d), 0], axis=0)

    # f = [zeros(d, 1); -0.5];
    f = np.concatenate([np.zeros(d, 1), -0.5], axis=0)

    # y   = @(lambda) (A'*A + lambda * D) \ (A'*b - lambda * f);
    def y(lam): return np.linalg.inv(A.T@A + lam * D) @ (A.T @ b - lam * f)
    # phi = @(lambda) y(lambda)' * D * y(lambda) + 2 * f' * y(lambda);
    def phi(lam): return y[lam].T @ D @ y[lam] + 2 * f.T @ y[lam]

    # eigDAA  = eig(D, A'*A);
    eigDAA = sp.linalg.eig(D, A.T @ A)
    # lambda1 = eigDAA(end);
    lambda1 = eigDAA[-1]

    # a1 = -1/lambda1;
    a1 = -1 / lambda1
    # a2 = 1000;
    a2 = 1000

    # epsAbs  = 1e-5;
    epsAbs = 1e-5
    # epsStep = 1e-5;
    epsStep = 1e-5

    # warning off;
    # while (a2 - a1 >= epsStep || ( abs( phi(a1) ) >= epsAbs && abs( phi(a2) )  >= epsAbs ) )
    while (a2 - a1) >= epsStep or (np.abs(phi[a1]) >= epsStep and np.abs(phi[a2] >= epsAbs)):
        #     c = (a1 + a2)/2;
        c = (a1 + a2) / 2
    #     if ( phi(c) == 0 )
        if (phi[c] == 0):
            break
    #        break;
        elif (phi[a1]@phi[c] < 0):
            a2 = c
    #     elseif ( phi(a1)*phi(c) < 0 )
    #        a2 = c;
        else:
            a1 = c
    #     else
    #        a1 = c;
    #     end
    # end
    # warning on;

    # estimatedLocation      = y(c);
    estimatedLocation = y[c]
    # estimatedLocation(end) = [];
    estimatedLocation[-1] = 0

    # totalError = sum(abs(sqrt(sum(bsxfun(@minus, anchors', estimatedLocation).^2)) - distances(:)'))
    totalError = 0
    return estimatedLocation, totalError
