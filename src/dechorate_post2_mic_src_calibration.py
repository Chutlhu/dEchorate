import h5py
import numpy as np
import scipy as sp
import pandas as pd
import scipy.signal as sg

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from tqdm import tqdm
from sklearn import manifold
from scipy.optimize import least_squares


from src.utils.file_utils import save_to_matlab

Fs = 48000 # Sampling frequency
T = 24     # temperature
speed_of_sound = 331.3 + 0.606 * T # speed of sound
L = int(0.5*Fs) # max length of the filter


def compute_distances_from_rirs(path_to_anechoic_dataset_rir, dataset):
    f_rir = h5py.File(path_to_anechoic_dataset_rir, 'r')

    all_src_ids = np.unique(dataset['src_id'])
    all_src_ids = all_src_ids[~np.isnan(all_src_ids)]
    all_mic_ids = np.unique(dataset['mic_id'])
    all_mic_ids = all_mic_ids[~np.isnan(all_mic_ids)]

    I = len(all_mic_ids)
    J = len(all_src_ids)


    mics_pos = np.zeros([3, I])
    srcs_pos = np.zeros([3, J])

    tofs_simulation = np.zeros([I, J])
    toes_rir = np.zeros([I, J])
    tofs_rir = np.zeros([I, J])
    toas_rir = np.zeros([I, J])

    rirs = np.zeros([L, I*J])
    direct_path_positions = np.zeros([I*J])
    ij = 0
    for j in tqdm(range(J)):
        for i in range(I):

            # find recording correspondent to mic i and src j
            entry = dataset.loc[(dataset['src_id'] == j+1) & (dataset['mic_id'] == i+1)]

            assert len(entry) == 1

            wavefile = entry['filename'].values[0]

            rir = f_rir['rir/%s/%d' % (wavefile, i)][()]
            rir = np.abs(rir).squeeze()
            rir = rir/np.max(rir)
            rirs[:, ij] = rir

            peaks, _ = sg.find_peaks(
                rir, height=0.2, distance=50, width=2, prominence=0.6)

            # plt.plot(rir)
            # plt.plot(peaks, rir[peaks], "x")
            # plt.plot(np.zeros_like(rir), "--", color="gray")
            # plt.show()

            # compute the theoretical distance
            mic_pos = [entry['mic_pos_x'].values, entry['mic_pos_y'].values, entry['mic_pos_z'].values]
            mics_pos[:, i] = np.array(mic_pos).squeeze()

            src_pos = [entry['src_pos_x'].values, entry['src_pos_y'].values, entry['src_pos_z'].values]
            srcs_pos[:, j] = np.array(src_pos).squeeze()

            d = np.linalg.norm(mics_pos[:, i] - srcs_pos[:, j])
            tof_geom = d / speed_of_sound

            # extract the time of arrival from the RIR
            direct_path_positions[ij] = np.min(peaks)
            recording_offset = f_rir['delay/%s/%d' % (wavefile, i)][()]
            # for recording with source j=5, the loopback is empty => wrong offset
            if j == 5:
                recording_offset = 6444

            toa = direct_path_positions[ij]/Fs

            tofs_simulation[i, j] = tof_geom
            toes_rir[i, j] = recording_offset/Fs
            toas_rir[i, j] = toa
            tofs_rir[i, j] = toa - recording_offset/Fs

            # try:
            #     assert tofs_rir[i, j] > 0
            #     assert np.abs(tofs_simulation[i,j] - tofs_rir[i, j]) * speed_of_sound < 0.10
            #     assert np.linalg.norm(rir[:recording_offset], ord=np.inf) < 0.1
            # except:
            #     # plot RIR, theoretical distance and peak-picking
            #     fig, ax = plt.subplots()
            #     newax1 = ax.twiny()
            #     newax2 = ax.twiny()
            #     fig.subplots_adjust(bottom=0.40)

            #     newax1.set_frame_on(True)
            #     newax2.set_frame_on(True)
            #     newax1.patch.set_visible(False)
            #     newax2.patch.set_visible(False)
            #     newax1.xaxis.set_ticks_position('bottom')
            #     newax2.xaxis.set_ticks_position('bottom')
            #     newax1.xaxis.set_label_position('bottom')
            #     newax2.xaxis.set_label_position('bottom')
            #     newax1.spines['bottom'].set_position(('outward', 40))
            #     newax2.spines['bottom'].set_position(('outward', 80))

            #     ax.plot(np.arange(len(rir)), rir)
            #     ax.axvline(recording_offset)
            #     ax.axvline(tof_geom*Fs + recording_offset)
            #     newax1.plot(np.arange(len(rir))/Fs, rir)
            #     newax2.plot(np.arange(len(rir))/Fs*speed_of_sound, rir)

            #     ax.set_xlabel('Time [samples]')
            #     newax1.set_xlabel('Time [seconds]')
            #     newax2.set_xlabel('Distance [meter]')

            #     plt.show()

            ij += 1

    plt.imshow(rirs, extent=[0, I*J, 0, L], aspect='auto')
    for j in range(J):
        plt.axvline(j*30, color='C7')
    plt.axhline(y=L-recording_offset, label='Time of Emission')
    plt.scatter(np.arange(I*J)+0.5, L-direct_path_positions, c='C1', label='Peak Picking')
    plt.scatter(np.arange(I*J)+0.5, L - recording_offset - tofs_simulation.T.flatten()*Fs, c='C2', label='Pyroom')
    plt.tight_layout()
    plt.legend()
    plt.savefig('./reports/figures/rir_skyline.pdf')
    plt.show()

    return tofs_simulation, tofs_rir, mics_pos, srcs_pos

def nlls_mds(D, init):
    X = init['X']
    A = init['A']
    dim, I = X.shape
    dim, J = A.shape
    assert D.shape == (I, J)

    def fun(xXA, I, J):
        xXA = xXA.reshape(3, I+J)
        X = xXA[:, :I]
        A = xXA[:, I:]
        cost = 0
        for i in range(I):
            for j in range(J):
                cost += (np.linalg.norm(X[:, i] - A[:, j]) - D[i, j])**2
        return cost

    x0 = np.concatenate([X, A], axis=1).flatten()
    res = least_squares(fun, x0, args=(I, J))
    print(res)
    solution = res.x
    solution = solution.reshape(3, I+J)
    X = solution[:, :I]
    A = solution[:, I:]
    return X, A

def crcc_mds(sqT, init):
    X = init['X']
    A = init['A']
    # [M, K] = size(sqT)
    I, J = sqT.shape
    # convert to squared "distances"
    T = sqT ** 2
    # T   = bsxfun(@minus, T, T(: , 1))
    # T   = bsxfun(@minus, T, T(1, : ))
    T = T - T[1, 1]
    # T = T(2: end, 2: end)
    T = T[1:, 1:]
    # D = (sqT * c). ^ 2
    D = sqT ** 2
    # [U, Sigma, V] = svd(T)
    U, Sigma, V = np.linalg.svd(T)
    Sigma = np.diag(Sigma)
    assert np.allclose(U[:, :Sigma.shape[0]] @ Sigma @ V, T)
    # Sigma = Sigma(1: 3, 1: 3)
    Sigma = Sigma[:3, :3]
    # U     = U(:, 1: 3)
    U = U[:, :3]
    # V     = V(:, 1: 3)
    V = V[:, :3]
    # Assume we know the distance between the first sensor and the first
    # microphone. This is realistic.
    # a1 = sqT(1, 1) * c
    a1 = sqT[1, 1]
    # function C = costC2(C, U, Sigma, V, D, a1)
    def edm(X, Y):
        # norm_X2 = sum(X. ^ 2);
        norm_X2 = np.sum(X ** 2, axis=0)[None, :]
        # norm_Y2 = sum(Y. ^ 2);
        norm_Y2 = np.sum(Y ** 2, axis=0)[None, :]
        # D = bsxfun(@plus, norm_X2', norm_Y2) - 2*X'*Y;
        D = norm_X2.T + norm_Y2 - 2 * X.T @ Y
        return D

    def fun(C, U, Sigma, V, D, a1):
        C = C.reshape([3,3])
        # X_tilde = (U*C)'
        X_tilde = (U @ C).T
        # Y_tilde = -1/2*inv(C)*Sigma*V'
        Y_tilde = -1/2 * np.linalg.inv(C) @ Sigma @ V.T
        # X = [[0 0 0]' X_tilde]
        X = np.concatenate([np.zeros([3, 1]), X_tilde], axis=1)
        # Y = [[0 0 0]' Y_tilde]
        Y = np.concatenate([np.zeros([3, 1]), Y_tilde], axis=1)
        # Y(1, :) = Y(1, : ) + a1
        Y[0, :] = Y[0, :] + a1
        # C = norm(edm(X, Y) - D, 'fro') ^ 2
        cost = np.linalg.norm(edm(X, Y) - D)**2
        return cost

    _, c0, _ = np.linalg.svd(edm(X, A))
    c0 = np.diag(c0[:3]).flatten()
    res = sp.optimize.minimize(fun, c0, args=(U, Sigma, V, D, a1), options={'disp':True})
    C = res.x.reshape([3,3])

    # tilde_R = (U*C)'
    R_tilde = (U@C).T
    # tilde_S = -1/2 * C\(Sigma*V')
    S_tilde = -1/2 * np.linalg.inv(C) @ Sigma @ V.T
    # R = [[0 0 0]' tilde_R]
    R = np.concatenate([np.zeros([3, 1]), R_tilde], axis=1)
    # This doesn't work for some reason(S)!!!
    # tilde_S(1, :) = tilde_S(1, : ) + a1
    S = np.concatenate([np.zeros([3, 1]), S_tilde], axis=1)
    # Y(1, :) = Y(1, : ) + a1
    S[0, :] = S[0, :] + a1
    # S = [[a1 0 0]' tilde_S]
    # D = edm([R S], [R S])
    return R, S

if __name__ == "__main__":
    dataset_dir = './data/dECHORATE/'
    path_to_processed = './data/processed/'

    path_to_anechoic_dataset_rir = path_to_processed + 'anechoic_rir_data.hdf5'

    path_to_database = dataset_dir + 'annotations/dECHORATE_database.csv'
    dataset = pd.read_csv(path_to_database)
    # select dataset with anechoic entries
    anechoic_dataset_chirp = dataset.loc[
          (dataset['room_rfl_floor'] == 0)
        & (dataset['room_rfl_ceiling'] == 0)
        & (dataset['room_rfl_west'] == 0)
        & (dataset['room_rfl_east'] == 0)
        & (dataset['room_rfl_north'] == 0)
        & (dataset['room_rfl_south'] == 0)
        & (dataset['src_signal'] == 'chirp')
    ]

    ## COMPUTE MICROPHONES-SOURCE DISTANCES
    tofs_simulation, tofs_rir, mics_pos, srcs_pos = compute_distances_from_rirs(
        path_to_anechoic_dataset_rir, anechoic_dataset_chirp)

    # plt.imshow(tofs_rir)
    # plt.show()

    save_to_matlab(path_to_processed + 'src_mic_dist.mat', tofs_rir)

    # ## MULTIDIMENSIONAL SCALING
    # # nonlinear least square problem with good initialization
    X = mics_pos
    A = srcs_pos
    # D = tofs_rir * speed_of_sound  # tofs_rir
    D = tofs_simulation * speed_of_sound
    # mics_pos_est, srcs_pos_est = nlls_mds(D, init={'X': X, 'A': A})
    # mics_pos_est, srcs_pos_est = crcc_mds(D, init={'X': X, 'A': A})
    # np.save(path_to_processed + 'mics_pos_est_nlls.npy', mics_pos_est)
    # np.save(path_to_processed + 'srcs_pos_est_nlls.npy', srcs_pos_est)
    mics_pos_est = np.load(path_to_processed + 'mics_pos_est_nlls.npy')
    srcs_pos_est = np.load(path_to_processed + 'srcs_pos_est_nlls.npy')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(mics_pos[0, :], mics_pos[1, :], mics_pos[2, :], marker='o', label='mics')
    ax.scatter(srcs_pos[0, :], srcs_pos[1, :], srcs_pos[2, :], marker='o', label='srcs')
    ax.scatter(mics_pos_est[0, :], mics_pos_est[1, :], mics_pos_est[2, :], marker='x', label='mics')
    ax.scatter(srcs_pos_est[0, :], srcs_pos_est[1, :], srcs_pos_est[2, :], marker='x', label='mics')
    plt.show()
    pass
