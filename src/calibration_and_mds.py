import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize

from src import constants
from src.utils.mds_utils import edm
from src.utils.dsp_utils import envelope, normalize



def nlls_mds(D, X, A, thr_mic=0.05, thr_src=0.05):
    dim, I = X.shape
    dim, J = A.shape
    assert D.shape == (I, J)

    def fun(xXA, I, J, D):
        X = xXA[:3*I].reshape(3, I)
        A = xXA[3*I:].reshape(3, J)

        cost = np.linalg.norm((edm(X, A) - D))**2

        return cost

    x0 = np.concatenate([X.flatten(), A.flatten()])
    ub = np.zeros_like(x0)
    ub[:] = np.inf
    lb = -ub
    # sources in +-5 cm from the guess
    dims_slacks = [thr_src, thr_src, thr_src]
    for j in range(J):
        for d in range(dim):
            ub[x0 == A[d, j]] = A[d, j] + dims_slacks[d]
            lb[x0 == A[d, j]] = A[d, j] - dims_slacks[d]
    # micros in +-5 cm from the guess
    dims_slacks = [thr_mic, thr_mic, thr_mic]
    for i in range(I):
        for d in range(dim):
            ub[x0 == X[d, i]] = X[d, i] + dims_slacks[d]
            lb[x0 == X[d, i]] = X[d, i] - dims_slacks[d]


    # set the origin in speaker 1
    bounds = sp.optimize.Bounds(lb, ub, keep_feasible=True)
    constraints = None
    # constraints = sp.optimize.LinearConstraint(A, lb, ub)

    res = sp.optimize.minimize(
            fun, x0, args=(I, J, D), bounds=bounds, constraints=constraints,
            options={'maxiter': 10e3, 'maxfev': 100e3})

    print('Optimization')
    print('message', res.message)
    print('nit', res.nit)
    print('nfev', res.nfev)
    print('success', res.success)
    print('fun', res.fun)
    sol = res.x
    # solution = solution.reshape(3, I+J)
    # X = solution[:, :I]
    # A = solution[:, I:]
    X = sol[:3*I].reshape(3, I)
    A = sol[3*I:].reshape(3, J)
    return X, A


def nlls_mds_images(Ddp, De_c, De_f, X, A, thr_mic=0.10, thr_src=.2):
    _, _, Rz = constants['room_size']

    dim, I = X.shape
    dim, J = A.shape
    assert Ddp.shape == (I,J) == De_c.shape == De_f.shape

    def fun_e1(xXA, I, J, Ddp, De_c, De_f):
        X = xXA[:3*I].reshape(3, I)
        A = xXA[3*I:].reshape(3, J)

        Ae_c = A.copy()
        Ae_f = A.copy()
        Ae_c[2, :] = 2*Rz - Ae_c[2, :]
        Ae_f[2, :] = -Ae_f[2, :]

        cost = np.linalg.norm((edm(X, A) - Ddp))**2         \
                + np.linalg.norm((edm(X, Ae_c) - De_c))**2  \
                + np.linalg.norm((edm(X, Ae_f) - De_f))**2

        return cost

    x0 = np.concatenate([X.flatten(), A.flatten()])
    ub = np.zeros_like(x0)
    ub[:] = np.inf
    lb = -ub
    # sources in +-5 cm from the guess
    dims_slacks = [thr_src, thr_src, thr_src]
    for j in range(J):
        for d in range(dim):
            ub[x0 == A[d, j]] = A[d, j] + dims_slacks[d]
            lb[x0 == A[d, j]] = A[d, j] - dims_slacks[d]
    # micros in +-5 cm from the guess
    dims_slacks = [thr_mic, thr_mic, thr_mic]
    for i in range(I):
        for d in range(dim):
            ub[x0 == X[d, i]] = X[d, i] + dims_slacks[d]
            lb[x0 == X[d, i]] = X[d, i] - dims_slacks[d]

    # set the origin in speaker 1
    bounds = sp.optimize.Bounds(lb, ub, keep_feasible=True)
    constraints = None
    # constraints = sp.optimize.LinearConstraint(A, lb, ub)

    res = sp.optimize.minimize(
        fun_e1, x0, args=(I, J, Ddp, De_c, De_f), bounds=bounds, constraints=constraints,
        options={'maxiter': 10e3, 'maxfev': 100e3, 'disp': True})

    print('Optimization')
    print('message', res.message)
    print('nit', res.nit)
    print('nfev', res.nfev)
    print('success', res.success)
    print('fun', res.fun)
    sol = res.x
    # solution = solution.reshape(3, I+J)
    # X = solution[:, :I]
    # A = solution[:, I:]
    X = sol[:3*I].reshape(3, I)
    A = sol[3*I:].reshape(3, J)
    return X, A


def nlls_mds_ceiling(Ddp, De_c, X, A, thr_mic=0.1, thr_src=.2):
    _, _, Rz = constants['room_size']

    dim, I = X.shape
    dim, J = A.shape
    assert Ddp.shape == (I, J) == De_c.shape

    def fun_e1(xXA, I, J, Ddp, De_c):
        X = xXA[:3*I].reshape(3, I)
        A = xXA[3*I:].reshape(3, J)

        Ae_c = A.copy()
        Ae_c[2, :] = 2*Rz - A[2, :]

        cost = np.linalg.norm((edm(X, A) - Ddp))**2   \
            + np.linalg.norm((edm(X, Ae_c) - De_c))**2

        return cost

    x0 = np.concatenate([X.flatten(), A.flatten()])
    ub = np.zeros_like(x0)
    ub[:] = np.inf
    lb = -ub
    # sources in +-5 cm from the guess
    dims_slacks = [thr_src, thr_src, thr_src]
    for j in range(J):
        for d in range(dim):
            ub[x0 == A[d, j]] = A[d, j] + dims_slacks[d]
            lb[x0 == A[d, j]] = A[d, j] - dims_slacks[d]
    # micros in +-5 cm from the guess
    dims_slacks = [thr_mic, thr_mic, thr_mic]
    for i in range(I):
        for d in range(dim):
            ub[x0 == X[d, i]] = X[d, i] + dims_slacks[d]
            lb[x0 == X[d, i]] = X[d, i] - dims_slacks[d]

    # set the origin in speaker 1
    bounds = sp.optimize.Bounds(lb, ub, keep_feasible=True)
    constraints = None
    # constraints = sp.optimize.LinearConstraint(A, lb, ub)

    res = sp.optimize.minimize(
        fun_e1, x0, args=(I, J, Ddp, De_c), bounds=bounds, constraints=constraints,
        options={'maxiter': 10e3, 'maxfev': 100e3})

    print('Optimization')
    print('message', res.message)
    print('nit', res.nit)
    print('nfev', res.nfev)
    print('success', res.success)
    print('fun', res.fun)
    sol = res.x
    # solution = solution.reshape(3, I+J)
    # X = solution[:, :I]
    # A = solution[:, I:]
    X = sol[:3*I].reshape(3, I)
    A = sol[3*I:].reshape(3, J)
    return X, A


def nlls_mds_array(D, X, A):
    dim, I = X.shape
    dim, J = A.shape
    assert D.shape == (I, J)

    nULA = np.zeros([3, 5])
    nULA[0, :] = np.array([0-3.25-5-4, 0-3.25-5, 0-3.25, 3.25, 3.25+10])/100

    def rotate_and_translate(LA, new_center, new_angle):
        # rotate
        th = np.deg2rad(new_angle)
        R = np.array([[[np.cos(th), -np.sin(th), 0],
                    [np.sin(th), np.cos(th),  0],
                    [0,          0,          1]]])
        nULA_rot = R@LA
        # translate
        nULA_tra = nULA_rot + new_center[:, None]
        return nULA_tra

    def fun(x, I, J, D):
        P = 6
        p = x[:4*P].reshape([4,P])
        arr_ang = p[0, :]    # I thetas
        arr_pos = p[1:4, :]  # I barycenters coordinates
        A = x[4*P:].reshape([3, J]) # J sources coordinates

        X = np.zeros([3, I])
        for i in range(P):
            X[:, i*5:(i+1)*5] = rotate_and_translate(nULA,arr_pos[:, i], arr_ang[i])

        cost = np.linalg.norm((edm(X, A) - D))**2

        return cost

    # x0 = np.concatenate([X.flatten(), A.flatten()])
    # ub = np.zeros_like(x0)
    # ub[:] = np.inf
    # lb = -ub
    # # sources in +-5 cm from the guess
    # dims_slacks = [thr, thr, thr]
    # for j in range(J):
    #     for d in range(dim):
    #         ub[x0 == A[d, j]] = A[d, j] + dims_slacks[d]
    #         lb[x0 == A[d, j]] = A[d, j] - dims_slacks[d]
    # # micros in +-5 cm from the guess
    # dims_slacks = [thr, thr, thr]
    # for i in range(I):
    #     for d in range(dim):
    #         ub[x0 == X[d, i]] = X[d, i] + dims_slacks[d]
    #         lb[x0 == X[d, i]] = X[d, i] - dims_slacks[d]

    # set the origin in speaker 1
    # bounds = sp.optimize.Bounds(lb, ub, keep_feasible=True)
    bounds = None
    constraints = None
    # constraints = sp.optimize.LinearConstraint(A, lb, ub)

    path_to_positions = './data/dECHORATE/positions.csv'
    audio_scene_positions = pd.read_csv(path_to_positions)

    mic_bar_pos = audio_scene_positions.loc[audio_scene_positions['type'] == 'array']
    mic_theta = np.array(mic_bar_pos['theta'])
    mic_bar_pos = np.vstack([mic_bar_pos['x'], mic_bar_pos['y'], mic_bar_pos['2.353']])
    x0 = np.concatenate([mic_theta.flatten(), mic_bar_pos.flatten(), A.flatten()])

    res = sp.optimize.minimize(
        fun, x0, args=(I, J, D), bounds=bounds, constraints=constraints,
        options={'maxiter': 10e3, 'maxfev': 100e3})

    print('Optimization')
    print('message', res.message)
    print('nit', res.nit)
    print('nfev', res.nfev)
    print('success', res.success)
    print('fun', res.fun)
    x = res.x
    P = 6
    p = x[:4*P].reshape([4,P])
    arr_ang = p[0, :]    # I thetas
    arr_pos = p[1:4, :]  # I barycenters coordinates
    A = x[4*P:].reshape([3, J]) # J sources coordinates

    X = np.zeros([3, I])
    for i in range(P):
        X[:, i*5:(i+1)*5] = rotate_and_translate(nULA,arr_pos[:, i], arr_ang[i])
    return X, A


def nlls_mds_array_ceiling(D, De_c, X, A):
    dim, I = X.shape
    dim, J = A.shape
    assert D.shape == (I, J)

    Rz = constants['room_size'][2]

    nULA = np.zeros([3, 5])
    nULA[0, :] = np.array([0-3.25-5-4, 0-3.25-5, 0-3.25, 3.25, 3.25+10])/100

    def rotate_and_translate(LA, new_center, new_angle):
        # rotate
        th = np.deg2rad(new_angle)
        R = np.array([[[np.cos(th), -np.sin(th), 0],
                       [np.sin(th), np.cos(th),  0],
                       [0,          0,          1]]])
        nULA_rot = R@LA
        # translate
        nULA_tra = nULA_rot + new_center[:, None]
        return nULA_tra

    def fun(x, I, J, D, De_c):
        P = 6
        p = x[:4*P].reshape([4, P])
        arr_ang = p[0, :]    # I thetas
        arr_pos = p[1:4, :]  # I barycenters coordinates
        A = x[4*P:].reshape([3, J])  # J sources coordinates

        X = np.zeros([3, I])
        for i in range(P):
            X[:, i*5:(i+1)*5] = rotate_and_translate(nULA,
                                                     arr_pos[:, i], arr_ang[i])

        Ac = A.copy()
        Ac[2, :] = Rz  + (Rz - A[2, :])

        cost = np.linalg.norm((edm(X, A) - D))**2 \
            + np.linalg.norm((edm(X, Ac) - De_c))**2

        return cost

    bounds = None
    constraints = None

    path_to_positions = './data/dECHORATE/positions.csv'
    audio_scene_positions = pd.read_csv(path_to_positions)

    mic_bar_pos = audio_scene_positions.loc[audio_scene_positions['type'] == 'array']
    mic_theta = np.array(mic_bar_pos['theta'])
    mic_bar_pos = np.vstack(
        [mic_bar_pos['x'], mic_bar_pos['y'], mic_bar_pos['2.353']])
    x0 = np.concatenate(
        [mic_theta.flatten(), mic_bar_pos.flatten(), A.flatten()])

    res = sp.optimize.minimize(
        fun, x0, args=(I, J, D, De_c), bounds=bounds, constraints=constraints,
        options={'maxiter': 10e3, 'maxfev': 100e3})

    print('Optimization')
    print('message', res.message)
    print('nit', res.nit)
    print('nfev', res.nfev)
    print('success', res.success)
    print('fun', res.fun)
    x = res.x
    P = 6
    p = x[:4*P].reshape([4, P])
    arr_ang = p[0, :]    # I thetas
    arr_pos = p[1:4, :]  # I barycenters coordinates
    A = x[4*P:].reshape([3, J])  # J sources coordinates

    X = np.zeros([3, I])
    for i in range(P):
        X[:, i*5:(i+1)*5] = rotate_and_translate(nULA,
                                                 arr_pos[:, i], arr_ang[i])
    return X, A


def nlls_mds_array_images(D, De_c, De_f, X, A):
    dim, I = X.shape
    dim, J = A.shape
    assert D.shape == (I, J)

    Rz = constants['room_size'][2]

    nULA = np.zeros([3, 5])
    nULA[0, :] = np.array([0-3.25-5-4, 0-3.25-5, 0-3.25, 3.25, 3.25+10])/100

    def rotate_and_translate(LA, new_center, new_angle):
        # rotate
        th = np.deg2rad(new_angle)
        R = np.array([[[np.cos(th), -np.sin(th), 0],
                       [np.sin(th), np.cos(th),  0],
                       [0,          0,          1]]])
        nULA_rot = R@LA
        # translate
        nULA_tra = nULA_rot + new_center[:, None]
        return nULA_tra

    def fun(x, I, J, D, De_c, De_f):
        P = 6
        p = x[:4*P].reshape([4, P])
        arr_ang = p[0, :]    # I thetas
        arr_pos = p[1:4, :]  # I barycenters coordinates
        A = x[4*P:].reshape([3, J])  # J sources coordinates

        X = np.zeros([3, I])
        for i in range(P):
            X[:, i*5:(i+1)*5] = rotate_and_translate(nULA,
                                                     arr_pos[:, i], arr_ang[i])

        Ac = A.copy()
        Af = A.copy()
        Ac[2, :] = 2*Rz - A[2, :]
        Af[2, :] = -A[2, :]


        cost = np.linalg.norm((edm(X, A) - D))**2 \
            + np.linalg.norm((edm(X, Ac) - De_c))**2  \
            + np.linalg.norm((edm(X, Af) - De_f))**2


        return cost

    bounds = None
    constraints = None

    path_to_positions = './data/dECHORATE/positions.csv'
    audio_scene_positions = pd.read_csv(path_to_positions)

    mic_bar_pos = audio_scene_positions.loc[audio_scene_positions['type'] == 'array']
    mic_theta = np.array(mic_bar_pos['theta'])
    mic_bar_pos = np.vstack(
        [mic_bar_pos['x'], mic_bar_pos['y'], mic_bar_pos['2.353']])
    x0 = np.concatenate(
        [mic_theta.flatten(), mic_bar_pos.flatten(), A.flatten()])

    res = sp.optimize.minimize(
        fun, x0, args=(I, J, D, De_c, De_f), bounds=bounds, constraints=constraints,
        options={'maxiter': 10e3, 'maxfev': 100e3})

    print('Optimization')
    print('message', res.message)
    print('nit', res.nit)
    print('nfev', res.nfev)
    print('success', res.success)
    print('fun', res.fun)
    x = res.x
    P = 6
    p = x[:4*P].reshape([4, P])
    arr_ang = p[0, :]    # I thetas
    arr_pos = p[1:4, :]  # I barycenters coordinates
    A = x[4*P:].reshape([3, J])  # J sources coordinates

    X = np.zeros([3, I])
    for i in range(P):
        X[:, i*5:(i+1)*5] = rotate_and_translate(nULA,
                                                 arr_pos[:, i], arr_ang[i])
    return X, A



def nlls_mds_array_with_rir_manifold(D, X, A, rirs_manifold):
    dim, I = X.shape
    dim, J = A.shape
    assert D.shape == (I, J)

    c = constants['speed_of_sound']
    Fs = constants['Fs']
    offset = constants['recording_offset']

    # smooth the manifold
    mani = np.zeros_like(rirs_manifold)
    for ij in range(mani.shape[1]):
        mani[:, ij] = 1-normalize(envelope(rirs_manifold[:, ij]))

    nULA = np.zeros([3, 5])
    nULA[0, :] = np.array([0-3.25-5-4, 0-3.25-5, 0-3.25, 3.25, 3.25+10])/100

    def rotate_and_translate(LA, new_center, new_angle):
        # rotate
        th = np.deg2rad(new_angle)
        R = np.array([[[np.cos(th), -np.sin(th), 0],
                       [np.sin(th), np.cos(th),  0],
                       [0,          0,          1]]])
        nULA_rot = R@LA
        # translate
        nULA_tra = nULA_rot + new_center[:, None]
        return nULA_tra

    def fun(x, I, J, D, mani, c):
        P = 6
        p = x[:4*P].reshape([4, P])
        arr_ang = p[0, :]    # I thetas
        arr_pos = p[1:4, :]  # I barycenters coordinates
        A = x[4*P:].reshape([3, J])  # J sources coordinates

        X = np.zeros([3, I])
        for i in range(P):
            X[:, i*5:(i+1)*5] = rotate_and_translate(nULA,
                                                     arr_pos[:, i], arr_ang[i])

        # cost = np.linalg.norm((edm(X, A) - D))**2
        E = np.round((edm(X, A)/c) * Fs + offset).astype(int).T.flatten()

        # for i, e in enumerate(E):
        #     plt.scatter(e, mani[e, i])
        #     plt.plot(mani[:, i])
        #     plt.show()
        #     input()
        cost = np.sum(mani[E, :])
        return cost

    # x0 = np.concatenate([X.flatten(), A.flatten()])
    # ub = np.zeros_like(x0)
    # ub[:] = np.inf
    # lb = -ub
    # # sources in +-5 cm from the guess
    # dims_slacks = [thr, thr, thr]
    # for j in range(J):
    #     for d in range(dim):
    #         ub[x0 == A[d, j]] = A[d, j] + dims_slacks[d]
    #         lb[x0 == A[d, j]] = A[d, j] - dims_slacks[d]
    # # micros in +-5 cm from the guess
    # dims_slacks = [thr, thr, thr]
    # for i in range(I):
    #     for d in range(dim):
    #         ub[x0 == X[d, i]] = X[d, i] + dims_slacks[d]
    #         lb[x0 == X[d, i]] = X[d, i] - dims_slacks[d]

    # set the origin in speaker 1
    # bounds = sp.optimize.Bounds(lb, ub, keep_feasible=True)
    bounds = None
    constraints = None
    # constraints = sp.optimize.LinearConstraint(A, lb, ub)

    path_to_positions = './data/dECHORATE/positions.csv'
    audio_scene_positions = pd.read_csv(path_to_positions)

    mic_bar_pos = audio_scene_positions.loc[audio_scene_positions['type'] == 'array']
    mic_theta = np.array(mic_bar_pos['theta'])
    mic_bar_pos = np.vstack(
        [mic_bar_pos['x'], mic_bar_pos['y'], mic_bar_pos['2.353']])
    x0 = np.concatenate(
        [mic_theta.flatten(), mic_bar_pos.flatten(), A.flatten()])

    res = sp.optimize.minimize(
        fun, x0, args=(I, J, D, mani, c), bounds=bounds, constraints=constraints,
        options={'maxiter': 10e3, 'maxfev': 100e3})

    print('Optimization')
    print('message', res.message)
    print('nit', res.nit)
    print('nfev', res.nfev)
    print('success', res.success)
    print('fun', res.fun)
    x = res.x
    P = 6
    p = x[:4*P].reshape([4, P])
    arr_ang = p[0, :]    # I thetas
    arr_pos = p[1:4, :]  # I barycenters coordinates
    A = x[4*P:].reshape([3, J])  # J sources coordinates

    X = np.zeros([3, I])
    for i in range(P):
        X[:, i*5:(i+1)*5] = rotate_and_translate(nULA,
                                                 arr_pos[:, i], arr_ang[i])
    return X, A


# def crcc_mds(Dobs, Xinit, Ainit):

#     Nd, Md = Dobs.shape
#     Dx, Nx = Xinit.shape
#     Da, Ma = Ainit.shape
#     assert Dx == Da == 3
#     assert Nx == Nd
#     assert Md == Ma
#     N = Nx
#     M = Md
#     D = Dx

#     # # Preparte observation
#     # Dt = Dobs**2 - Dobs[0:1, :]**2 - Dobs[:, 0:1]**2 + Dobs[0, 0]**2
#     # Dt = Dt[1:, 1:]

#     # Prepare initialization
#     X = Xinit
#     A = Ainit
#     # center wrt the first entry
#     A = A - A[:, 0:1] # correspond to \texttt{A} in the paper
#     X = X - X[:, 0:1] # correspond to \texttt{X} in the paper
#     # Crocco wants the matrix as DxM and DxN
#     # and remove the 1rst row
#     Xt = X[:, 1:].T
#     At = A[:, 1:].T
#     Dinit = (-2 * Xt @ At.T)
#     assert Dinit.shape == Dt.shape
#     assert np.allclose(Dinit, Dt)
#     Ui, Vi, Whi = np.linalg.svd(Dinit)
#     Cinit = np.linalg.pinv(Ui[:, :3]) @ Xt

#     # D = edm(X, A)**0.5
#     # for i in range(N):
#     #     for j in range(M):
#     #         assert np.allclose(D[i, j]**2, np.linalg.norm(X[:, i] - A[:, j])**2)

#     U, V, Wh = np.linalg.svd(Dt)
#     # select only the first 3 eigenvalue
#     # imposing rank=3
#     V = np.diag(V[:3])
#     U = U[:N-1, :3]
#     Wh = Wh[:3, :M-1]
#     assert np.allclose(Dt, U@V@Wh)

#     # def f(x0, U, V, Wh, D):
#     #     C = x0[:9].reshape([3,3])
#     #     a00 = x0[-1]

#     #     cost = 0
#     #     UC = U @ C
#     #     UVW = U @ V @ Wh

#     #     for i in range(N-1):
#     #         for j in range(M-1):
#     #             uc_sq_sum = np.sum(UC[i, :])**2
#     #             uvw_sq_sum = np.sum(UVW[i, j])**2
#     #             _2uc_a00 = -2 * UC[i, 0]*a00
#     #             _dist = -D[i+1,j+1]**2 + D[0, j+1]**2
#     #             cost += (uc_sq_sum \
#     #                      + uvw_sq_sum \
#     #                      + _2uc_a00 \
#     #                      + _dist)**2
#     #     return cost
#     def fun(C, U, V, Wh, D, a1):
#         C = C.reshape([3, 3])
#         Xt = U @ C
#         At = (- 0.5 * np.linalg.inv(C) @ V @ Wh).T
#         Xt = np.concatenate([np.zeros([1, 3]), Xt], axis=0)
#         At = np.concatenate([np.zeros([1, 3]), At], axis=0)
#         At[0, :] = At[0, :] + a1
#         print(edm(Xt.T,At.T))
#         print(D)
#         1/0
#         cost = np.linalg.norm(edm(X, Y) - D)
#         return cost

#     x0 = np.concatenate([np.random.randn(3,3).flatten()])
#     print(fun(x0, U, V, Wh, Dobs, Ainit[0, 0]))

#     res = minimize(fun, x0, args=(U, V, Wh, Dobs, Ainit[0,0]), options={'disp': True})
#     print(res)


#     1/0
#     return X


def crcc_mds1(D, Xinit, Ainit):
    # [M, K] = size(sqT)
    I, J = D.shape
    dimx, Ix = Xinit.shape
    dima, Ja = Ainit.shape
    assert I == Ix
    assert J == Ja
    assert dimx == dima

    # acccording to Crocco notation
    Xinit = Xinit.T
    Ainit = Ainit.T
    assert edm(Xinit.T, Ainit.T).shape == D.shape
    I, J = D.shape

    Xt_init = (Xinit - Xinit[0:1, :])[1:, :]
    At_init = (Ainit - Ainit[0:1, :])[1:, :]
    Dt_init = -2 * Xt_init @ At_init.T

    ## FROM EDM TO Dtilde (centred and cropped)
    # convert to squared "distances"
    D2 = D ** 2
    D2 = D2 - D2[0:1, :] - D2[:, 0:1] + D2[0, 0]
    Dt = D2[1:, 1:]
    assert Dt.shape == (I-1, J-1)
    assert np.allclose(Dt, Dt_init)

    ## SVD
    # [U, Sigma, V] = svd(T)
    U, V, Wh = np.linalg.svd(Dt)
    V = np.diag(V[:3])
    U = U[:I-1, :3]
    Wh = Wh[:3, :J-1]
    assert np.allclose(Dt, U @ V @ Wh)

    # def fun(C, U, V, Wh, D, a1):
    #     C = C.reshape([3, 3])
    #     # X_tilde = (U*C)'
    #     X = U @ C.T
    #     # Y_tilde = -1/2*inv(C)*Sigma*V'
    #     A = -0.5 * np.linalg.inv(C) @ V @ Wh
    #     # X = [[0 0 0]' X_tilde]
    #     # Y = [[0 0 0]' Y_tilde]
    #     X = np.concatenate([np.zeros([3, 1]), X])
    #     A = np.concatenate([np.zeros([3, 1]), A])
    #     # Y(1, :) = Y(1, : ) + a1
    #     A[0, :] = A[0, :] + a1
    #     # C = norm(edm(X, Y) - D, 'fro') ^ 2
    #     return np.linalg.norm(edm(X,Y) - D)

    def fun(C, U, V, Wh, D, a):
        C = C.reshape([3, 3])
        UC = U@C
        UVWh = U @ V @ Wh
        cost = 0
        for i in range(1, I-1):
            for j in range(1, J-1):
                c = np.sum(UC[i, :]**2)
                c += np.sum(UVWh[i, j])
                c += -2*UC[i, 0]*a
                c += -D[i+1, j+1]**2 + D[0, j+1]**2
                cost = c**2
        return cost


    x0 = np.random.randn(3,3).flatten()
    res = sp.optimize.minimize(fun, x0, args=(U, V, Wh, D, Ainit[0, 0]), options={'disp': True})
    C = res.x.reshape([3, 3])

    Xt_est = U @ C
    At_est = -0.5 * (np.linalg.inv(C) @ V @ Wh).T

    Xt_est = np.concatenate([np.zeros([1, 3]), Xt_est])
    At_est = np.concatenate([np.zeros([1, 3]), At_est])
    At_est[0, 0] = Ainit[0, 0]

    print(edm(Xt_est.T, At_est.T))
    # # Assume we know the distance between the first sensor and the first
    # # microphone. This is realistic.
    # a1 = D[1, 1]

    # def fun(C, U, V, Wh, D, a1):
    #     C = C.reshape([3, 3])
    #     # X_tilde = (U*C)'
    #     X_tilde = (U @ C).T
    #     # Y_tilde = -1/2*inv(C)*Sigma*V'
    #     Y_tilde = - 1/2 * np.linalg.inv(C) @ V @ Wh
    #     # X = [[0 0 0]' X_tilde]
    #     X = np.concatenate([np.zeros([3, 1]), X_tilde], axis=1)
    #     # Y = [[0 0 0]' Y_tilde]
    #     Y = np.concatenate([np.zeros([3, 1]), Y_tilde], axis=1)
    #     # Y(1, :) = Y(1, : ) + a1
    #     Y[0, :] = Y[0, :] + a1
    #     # C = norm(edm(X, Y) - D, 'fro') ^ 2
    #     cost = np.linalg.norm(edm(X, Y) - D)**2
    #     return cost

    # _, c0, _ = np.linalg.svd(edm(X, A))
    # c0 = np.diag(c0[:3]).flatten()
    # res = sp.optimize.minimize(fun, c0, args=(U, V, Wh, D, a1), options={'disp': True})
    # C = res.x.reshape([3, 3])
    # 1/0

    # # tilde_R = (U*C)'
    # R_tilde = (U@C).T
    # # tilde_S = -1/2 * C\(Sigma*V')
    # S_tilde = -1/2 * np.linalg.inv(C) @ Sigma @ V.T
    # # R = [[0 0 0]' tilde_R]
    # R = np.concatenate([np.zeros([3, 1]), R_tilde], axis=1)
    # # This doesn't work for some reason(S)!!!
    # # tilde_S(1, :) = tilde_S(1, : ) + a1
    # S = np.concatenate([np.zeros([3, 1]), S_tilde], axis=1)
    # # Y(1, :) = Y(1, : ) + a1
    # S[0, :] = S[0, :] + a1
    # # S = [[a1 0 0]' tilde_S]
    # # D = edm([R S], [R S])
    # return R, S


# function[R, S, D] = unfold_crocco(sqT, c)
# %
# % [R, S, D] = unfold_crocco(sqT, c)
# %
# % Solves the multidimensional unfolding(MDU) problem using the method in
# % Marco Crocco, Alessio Del Bue, and Vittorio Murino: A Bilinear Approach to
# % the Position Self-Calibration of Multiple Sensors
# %
# %
# % INPUT:  T ... M by K matrix, where M is the number of microphones, and K
# %               the number of sources(acoustic events)
# T(i, j) is the
# %               propagation time between the i-th microphone and the j-th
# %               source
# %         c ... speed of sound
# %
# % OUTPUT: R ... (dim by m) Estimated microphone locations
# %         S ... (dim by k) Estimated source locations
# %         D ... ((m+k) by(m+k)) Resulting EDM
# %
# %
# % Author: Ivan Dokmanic, 2014

# [M, K] = size(sqT)

# T = (sqT * c). ^ 2
# % convert to squared "distances"
# T   = bsxfun(@minus, T, T(: , 1))
# % (*)
# T   = bsxfun(@minus, T, T(1, : ))
# T = T(2: end, 2: end)

# D = (sqT * c). ^ 2

# [U, Sigma, V] = svd(T)

# Sigma = Sigma(1: 3, 1: 3)
# U     = U(:, 1: 3)
# V     = V(:, 1: 3)

# % Assume we know the distance between the first sensor and the first
# % microphone. This is realistic.
# a1 = sqT(1, 1) * c

# opt = optimset('MaxfevEvals', 1e8, 'MaxIter', 1e6)

# MAX_ITER = 0
# [Cbest, costbest] = fminsearch(@(C) costC2(C, U, Sigma, V, D, a1), randn(3), opt)
# for i = 1:
#     MAX_ITER
#     i
#     [C, costval] = fminsearch(@(C) costC2(C, U, Sigma, V, D, a1), randn(3), opt)
#     if costval < costbest
#     costbest = costval
#     Cbest = C
#     end
# end
# C = Cbest

# tilde_R = (U*C)'
# tilde_S = -1/2 * C\(Sigma*V')

# R = [[0 0 0]' tilde_R]

# % This doesn't work for some reason(S)!!!
# tilde_S(1, :) = tilde_S(1, : ) + a1
# S = [[a1 0 0]' tilde_S]

# D = edm([R S], [R S])


# function C = costC2(C, U, Sigma, V, D, a1)

# X_tilde = (U*C)'
# Y_tilde = -1/2*inv(C)*Sigma*V'

# X = [[0 0 0]' X_tilde]

# Y = [[0 0 0]' Y_tilde]
# Y(1, :) = Y(1, : ) + a1

# C = norm(edm(X, Y) - D, 'fro') ^ 2
