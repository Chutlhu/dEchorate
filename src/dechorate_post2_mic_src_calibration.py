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
from src.utils.dsp_utils import envelope
from src import constants

from risotto import deconvolution as deconv

Fs = constants['Fs'] # Sampling frequency
recording_offset = constants['recording_offset']
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

            rir = f_rir['rir/%s/%d' % (wavefile, i)][()].squeeze()
            rir_abs = np.abs(rir)
            rir_abs = rir_abs/np.max(rir_abs)
            rirs[:, ij] = rir_abs

            # compute the theoretical distance
            mic_pos = [entry['mic_pos_x'].values, entry['mic_pos_y'].values, entry['mic_pos_z'].values]
            mics_pos[:, i] = np.array(mic_pos).squeeze()

            src_pos = [entry['src_pos_x'].values, entry['src_pos_y'].values, entry['src_pos_z'].values]
            srcs_pos[:, j] = np.array(src_pos).squeeze()

            d = np.linalg.norm(mics_pos[:, i] - srcs_pos[:, j])
            tof_geom = d / speed_of_sound

            # image wtr to the ceiling
            imag_src_pos_ceiling = srcs_pos[:, j].copy()
            imag_src_pos_ceiling[2] = 2.353 + (2.353 - imag_src_pos_ceiling[2])
            imag_src_pos_floor = srcs_pos[:, j].copy()
            imag_src_pos_floor[2] = -imag_src_pos_floor[2]
            d_ceiling = np.linalg.norm(imag_src_pos_ceiling - mics_pos[:, i])
            d_floor   = np.linalg.norm(imag_src_pos_floor - mics_pos[:, i])
            tof_geom_ceiling = d_ceiling / speed_of_sound
            tof_geom_floor = d_floor / speed_of_sound

            thr1 = int(recording_offset + Fs*(tof_geom + np.abs(np.min([tof_geom_ceiling,tof_geom_floor])-tof_geom)/2))
            thr2 = int(recording_offset + Fs*(np.max([tof_geom_ceiling, tof_geom_floor])) + 200)

            # direct path peak
            peaks, _ = sg.find_peaks(rir_abs, height=0.2, distance=50, width=2, prominence=0.6)
            dp_peak = np.min(peaks)
            # floor
            d_min_floor_ceiling = np.abs(tof_geom_ceiling - tof_geom_floor)*Fs
            peaks, _ = sg.find_peaks(
                rir_abs[thr1:thr2], height=0.2, distance=50, width=2, prominence=0.2)
            peaks = peaks + thr1

            # # direct path deconvolution
            # x = np.zeros_like(rir)
            # a = 30
            # x[:2*a] = rir[dp_peak-a:dp_peak+a]
            # y = rir.copy()
            # h = np.zeros_like(rir)
            # h_tmp = deconv.wiener_deconvolution(y, x)
            # h[a:] = h_tmp[:-a]
            # rir_abs = np.abs(h)


            # # if j == 7:
            plt.title('Source %d, array %d, microphone %d' % (j+1, i//5 + 1, i % 5+1))
            plt.plot(rir_abs)
            plt.plot(peaks, rir_abs[peaks], "x")
            plt.plot(dp_peak, rir_abs[dp_peak], "o")
            plt.axvline(x=recording_offset, color='C0', label='offset')
            plt.axvline(x=tof_geom*Fs+recording_offset, color='C0', label='direct')
            # plt.axvline(x=tof_geom_ceiling*Fs+recording_offset, color='C1', label='ceiling')
            plt.axvline(x=tof_geom_floor*Fs+recording_offset, color='C2', label='floor')
            plt.legend()
            plt.xlim([recording_offset-100, recording_offset+800])
            plt.show()

            # extract the time of arrival from the RIR
            direct_path_positions[ij] = dp_peak

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

    return rirs, recording_offset, tofs_simulation, tofs_rir, mics_pos, srcs_pos

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
    ub = np.zeros_like(x0)
    ub[:] = np.inf
    lb = -ub
    # sources in +-5 cm from the guess
    for j in range(J):
        for d in range(dim):
            ub[x0 == A[d, j]] = A[d, j] + 0.50
            lb[x0 == A[d, j]] = A[d, j] - 0.50
    # micros in +-5 cm from the guess
    for i in range(I):
        for d in range(dim):
            ub[x0 == X[d, i]] = X[d, i] + 0.50
            lb[x0 == X[d, i]] = X[d, i] - 0.50

    # set the origin in speaker 1
    bounds = sp.optimize.Bounds(lb, ub)
    res = sp.optimize.minimize(fun, x0, args=(I, J), bounds=bounds, options={'maxiter':10e3, 'maxfun':100e3})
    print('Optimization')
    print('message', res.message)
    print('nit', res.nit)
    print('nfev', res.nfev)
    print('success', res.success)
    print('fun', res.fun)
    solution = res.x
    solution = solution.reshape(3, I+J)
    X = solution[:, :I]
    A = solution[:, I:]
    return X, A

def edm(X, Y):
    # norm_X2 = sum(X. ^ 2);
    norm_X2 = np.sum(X ** 2, axis=0)[None, :]
    # norm_Y2 = sum(Y. ^ 2);
    norm_Y2 = np.sum(Y ** 2, axis=0)[None, :]
    # D = bsxfun(@plus, norm_X2', norm_Y2) - 2*X'*Y;
    D = norm_X2.T + norm_Y2 - 2 * X.T @ Y
    return D

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

    session_id = '000000' # '010000'
    path_to_anechoic_dataset_rir = path_to_processed + '%s_rir_data.hdf5' % session_id

    path_to_database = dataset_dir + 'annotations/dECHORATE_database.csv'
    dataset = pd.read_csv(path_to_database)
    # select dataset with entries according to session_id
    f, c, w, e, n, s = [int(i) for i in list(session_id)]
    anechoic_dataset_chirp = dataset.loc[
          (dataset['room_rfl_floor'] == f)
        & (dataset['room_rfl_ceiling'] == c)
        & (dataset['room_rfl_west'] == w)
        & (dataset['room_rfl_east'] == e)
        & (dataset['room_rfl_north'] == n)
        & (dataset['room_rfl_south'] == s)
        & (dataset['room_fornitures'] == False)
        & (dataset['src_signal'] == 'chirp')
    ]

    ## COMPUTE DIRECT PATH POSITIONS
    rirs, recording_offset, tofs_simulation, tofs_rir, mics_pos, srcs_pos = compute_distances_from_rirs(
        path_to_anechoic_dataset_rir, anechoic_dataset_chirp)

    L, IJ = rirs.shape
    D, I = mics_pos.shape
    D, J = srcs_pos.shape

    plt.imshow(rirs, extent=[0, I*J, 0, L], aspect='auto')
    for j in range(J):
        plt.axvline(j*30, color='C7')
    plt.axhline(y=L-recording_offset, label='Time of Emission')
    plt.scatter(np.arange(I*J)+0.5, L - recording_offset - tofs_rir.T.flatten()*Fs, c='C1', label='Peak Picking')
    plt.scatter(np.arange(I*J)+0.5, L - recording_offset - tofs_simulation.T.flatten()*Fs, c='C2', label='Pyroom')
    plt.tight_layout()
    plt.legend()
    plt.savefig('./reports/figures/rir_skyline.pdf')
    plt.show()

    save_to_matlab(path_to_processed + 'src_mic_dist.mat', tofs_rir)

    # ## MULTIDIMENSIONAL SCALING
    # # nonlinear least square problem with good initialization
    X = mics_pos
    A = srcs_pos[:, :4]
    # D = tofs_simulation * speed_of_sound
    Dedm = edm(X, A) ** (.5)
    Dtof = tofs_rir[:, :4] * speed_of_sound
    Dgeo = tofs_simulation[:, :4] * speed_of_sound
    assert np.allclose(Dedm, Dgeo)

    sp.io.savemat('./data/processed/calibration_data.mat', {'D':D, 'X':X, 'A':A})
    mics_pos_est, srcs_pos_est = nlls_mds(Dtof, init={'X': X, 'A': A})

    # mics_pos_est, srcs_pos_est = crcc_mds(D, init={'X': X, 'A': A})
    # # np.save(path_to_processed + 'mics_pos_est_nlls.npy', mics_pos_est)
    # # np.save(path_to_processed + 'srcs_pos_est_nlls.npy', srcs_pos_est)
    # mics_pos_est = np.load(path_to_processed + 'mics_pos_est_nlls.npy')
    # srcs_pos_est = np.load(path_to_processed + 'srcs_pos_est_nlls.npy')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[0, :], X[1, :], X[2, :], marker='o', label='mics init')
    ax.scatter(A[0, :], A[1, :], A[2, :], marker='o', label='srcs init')
    ax.scatter(mics_pos_est[0, :], mics_pos_est[1, :], mics_pos_est[2, :], marker='x', label='mics est')
    ax.scatter(srcs_pos_est[0, :], srcs_pos_est[1, :], srcs_pos_est[2, :], marker='x', label='srcs est')
    plt.legend()
    plt.show()

    new_tofs = (edm(mics_pos_est, srcs_pos_est) ** .5) / speed_of_sound

    # plt.imshow(rirs, extent=[0, I*J, 0, L], aspect='auto')
    for j in range(J):
        plt.axvline(j*30, color='C7')
    plt.axhline(y=L-recording_offset, label='Time of Emission')
    plt.scatter(np.arange(I*J)+0.5, L - recording_offset - tofs_rir.T.flatten()*Fs, c='C1', label='Peak Picking')
    plt.scatter(np.arange(I*J)+0.5, L - recording_offset - tofs_simulation.T.flatten()*Fs, c='C2', label='Pyroom')
    plt.scatter(np.arange(len(new_tofs.flatten()))+0.5, L - recording_offset - new_tofs.T.flatten()*Fs, c='C3', marker='X', label='After EDM')
    plt.tight_layout()
    plt.legend()
    plt.savefig('./reports/figures/rir_skyline_after_calibration.pdf')
    plt.show()

    # Blueprint 2D xz plane
    room_size = [5.543, 5.675, 2.353]
    plt.figure(figsize=(16, 9))
    plt.gca().add_patch(
        plt.Rectangle((0, 0),
                    room_size[0], room_size[2], fill=False,
                    edgecolor='g', linewidth=1)
    )

    plt.scatter(mics_pos_est[0, :], mics_pos_est[2, :], marker='X')
    plt.scatter(srcs_pos_est[0, :], srcs_pos_est[2, :], marker='v')
    # for i in range(I):
        # plt.text(mics_pos_est[0, i], mics_pos_est[2, i], '$%d$' %
        #         (i+33), fontdict={'fontsize': 8})
        # if i % 5 == 0:
        #     bar = np.mean(mics_pos_est[:, 5*i//5:5*(i//5+1)], axis=1)
        #     plt.text(bar[0]+0.1, bar[2]+0.1, '$arr_%d$ [%1.2f, %1.2f, %1.2f]' %
        #             (i//5 + 1, bar[0], bar[1], bar[2]), fontdict={'fontsize': 8})

    # for j in range(J):
    #     bar = srcs_pos_est[:, j]
    #     if j < 6:
    #         plt.text(bar[0], bar[2], '$dir_%d$ [%1.2f, %1.2f, %1.2f]' %
    #                 (j+1, bar[0], bar[2], bar[2]), fontdict={'fontsize': 8})
    #     else:
    #         plt.scatter(bar[0], bar[2], marker='o')
    #         plt.text(bar[0], bar[2], '$omn_%d$ [%1.2f, %1.2f, %1.2f]' %
                    # (j+1, bar[0], bar[1], bar[2]), fontdict={'fontsize': 8})
    plt.savefig('./reports/figures/cal_positioning2D_xz.pdf')
    plt.show()

    pass
