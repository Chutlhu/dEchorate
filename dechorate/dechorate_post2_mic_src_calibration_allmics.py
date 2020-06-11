import h5py
import numpy as np
import scipy as sp
import pandas as pd
import scipy.interpolate as intp
import scipy.signal as sg

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from tqdm import tqdm
from sklearn import manifold
from scipy.optimize import least_squares

from dechorate import constants

from dechorate.dataset import DechorateDataset, SyntheticDataset
from dechorate.calibration_and_mds import *

from dechorate.utils.file_utils import save_to_matlab, load_from_pickle, save_to_pickle
from dechorate.utils.dsp_utils import envelope, normalize
from dechorate.utils.mds_utils import edm

from risotto import deconvolution as deconv

ox, oy, oz = constants['offset_beacon']

Fs = constants['Fs'] # Sampling frequency
recording_offset = constants['recording_offset']
Rx, Ry, Rz = constants['room_size']
speed_of_sound = constants['speed_of_sound']  # speed of sound

dataset_dir = './data/dECHORATE/'
path_to_processed = './data/processed/'


def load_rirs(path_to_dataset_rir, dataset, K, dataset_id, mics_pos, srcs_pos):

    f_rir = h5py.File(path_to_dataset_rir, 'r')

    all_src_ids = np.unique(dataset['src_id'])
    all_src_ids = all_src_ids[~np.isnan(all_src_ids)]
    all_mic_ids = np.unique(dataset['mic_id'])
    all_mic_ids = all_mic_ids[~np.isnan(all_mic_ids)]

    I = len(all_mic_ids)
    J = len(all_src_ids)

    if mics_pos is None:
        mics_pos = np.zeros([3, I])
    if srcs_pos is None:
        srcs_pos = np.zeros([3, J])


    toa_sym = np.zeros([7, I, J])

    amp_sym = np.zeros([7, I, J])
    ord_sym = np.zeros([7, I, J])
    wal_sym = np.chararray([7, I, J])

    L = int(0.5*Fs) # max length of the filter
    rirs = np.zeros([L, I, J])

    for j in tqdm(range(J)):
        for i in range(I):

            # find recording correspondent to mic i and src j
            entry = dataset.loc[(dataset['src_id'] == j+1) & (dataset['mic_id'] == i+1)]

            assert len(entry) == 1

            wavefile = entry['filename'].values[0]

            rir = f_rir['rir/%s/%d' % (wavefile, i)][()].squeeze()

            rir = rir/np.max(np.abs(rir))
            rirs[:, i, j] = np.abs(rir)

            # compute the theoretical distance
            if np.allclose(mics_pos[:, i], 0):
                print('mic', i, 'from manual annotation')
                mic_pos = [entry['mic_pos_x'].values, entry['mic_pos_y'].values, entry['mic_pos_z'].values]
                # apply offset
                mic_pos[0] = mic_pos[0] + constants['offset_beacon'][0]
                mic_pos[1] = mic_pos[1] + constants['offset_beacon'][1]
                mic_pos[2] = mic_pos[2] + constants['offset_beacon'][2]

                mics_pos[:, i] = np.array(mic_pos).squeeze()

            if np.allclose(srcs_pos[:, j], 0):
                print('src', j, 'from manual annotation')
                src_pos = [entry['src_pos_x'].values, entry['src_pos_y'].values, entry['src_pos_z'].values]
                # apply offset
                src_pos[0] = src_pos[0] + constants['offset_beacon'][0]
                src_pos[1] = src_pos[1] + constants['offset_beacon'][1]
                src_pos[2] = src_pos[2] + constants['offset_beacon'][2]
                srcs_pos[:, j] = np.array(src_pos).squeeze()

            synth_dset = SyntheticDataset()
            synth_dset.set_room_size(constants['room_size'])
            synth_dset.set_dataset(dataset_id, absb=1, refl=0)
            synth_dset.set_c(speed_of_sound)
            synth_dset.set_k_order(1)
            synth_dset.set_k_reflc(7)
            synth_dset.set_mic(mics_pos[0, i], mics_pos[1, i], mics_pos[2, i])
            synth_dset.set_src(srcs_pos[0, j], srcs_pos[1, j], srcs_pos[2, j])
            amp, tau, wall, order, gen = synth_dset.get_note()

            toa_sym[:, i, j] = tau
            amp_sym[:, i, j] = amp
            wal_sym[:, i, j] = wall
            ord_sym[:, i, j] = order

    return rirs, toa_sym, mics_pos, srcs_pos


def iterative_calibration(dataset_id, mics_pos, srcs_pos, K, toa_peak):

    refl_order = constants['refl_order_pyroom']
    curr_reflectors = constants['refl_order_calibr'][:K+1]

    d = constants['datasets'].index(dataset_id)
    path_to_dataset_rir = path_to_processed + '%s_rir_data.hdf5' % dataset_id
    path_to_database = dataset_dir + 'annotations/dECHORATE_database.csv'
    dataset = pd.read_csv(path_to_database)
    # select dataset with entries according to session_id
    f, c, w, e, n, s = [int(i) for i in list(dataset_id)]

    dataset = dataset.loc[
        (dataset['room_rfl_floor'] == f)
        & (dataset['room_rfl_ceiling'] == c)
        & (dataset['room_rfl_west'] == w)
        & (dataset['room_rfl_east'] == e)
        & (dataset['room_rfl_north'] == n)
        & (dataset['room_rfl_south'] == s)
        & (dataset['room_fornitures'] == False)
        & (dataset['src_signal'] == 'chirp')
        & (dataset['src_id'] < 5)
    ]


    # LOAD MEASURED RIRs
    # and COMPUTED PYROOM ANNOTATION
    rirs, toa_sym, mics_pos, srcs_pos = load_rirs(path_to_dataset_rir, dataset, K, dataset_id, mics_pos, srcs_pos)

    print(toa_peak.shape, toa_sym.shape)
    assert toa_peak.shape == toa_sym.shape
    assert toa_peak.shape[1] == mics_pos.shape[1]
    assert toa_peak.shape[2] == srcs_pos.shape[1]

    L, I, J = rirs.shape
    D, I = mics_pos.shape
    D, J = srcs_pos.shape

    rirs_skyline = rirs.transpose([0, 2, 1]).reshape([L, I*J])

    plt.imshow(rirs_skyline, extent=[0, I*J, 0, L], aspect='auto')
    for j in range(J):
            plt.axvline(j*30, color='C7')
    plt.axhline(y=L-recording_offset, label='Time of Emission')

    for k in range(K+1):
        print(curr_reflectors)
        wall = curr_reflectors[k]
        r = refl_order.index(wall)
        plt.scatter(np.arange(I*J)+0.5, L - recording_offset - toa_peak[r,:,:].T.flatten()*Fs,
                    c='C%d'%(k+2), marker='x', label='Peak Picking')
        plt.scatter(np.arange(I*J)+0.5, L - recording_offset - toa_sym[r, :, :].T.flatten()*Fs,
                    marker='o', facecolors='none', edgecolors='C%d'%(k+2), label='Pyroom')

    plt.tight_layout()
    plt.legend()
    plt.title('RIR SKYLINE K = %d' % K)
    plt.savefig('./reports/figures/rir_skyline.pdf')
    plt.show()
    # plt.close()

    # ## MULTIDIMENSIONAL SCALING
    # select sub set of microphones and sources

    # # nonlinear least square problem with good initialization
    X = mics_pos
    A = srcs_pos

    Dgeo = edm(X, A)
    if K == 0:
        curr_refl_name = 'd'
        r = refl_order.index(curr_refl_name)
        Dtoa = toa_peak[r, :I, :J] * speed_of_sound
        Dsym = toa_sym[r, :I, :J] * speed_of_sound
    if K == 1:
        Dtoa = toa_peak[0, :I, :J] * speed_of_sound
        Dsym = toa_sym[0, :I, :J] * speed_of_sound
        wall = curr_reflectors[1]
        r = refl_order.index(wall)
        De_c = toa_peak[r, :I, :J] * speed_of_sound
    # if K == 2:
    #     Dobs = toa_peak[0, :I, :J] * speed_of_sound
    #     Dsym = toa_sym[0, :I, :J] * speed_of_sound
    #     wall = curr_reflectors[1]
    #     r = refl_order.index(wall)
    #     print(wall, r)
    #     De_c = toa_peak[r, :I, :J] * speed_of_sound
    #     wall = curr_reflectors[2]
    #     r = refl_order.index(wall)
    #     print(wall, r)
    #     De_f = toa_peak[r, :I, :J] * speed_of_sound

    assert np.allclose(Dgeo, Dsym)

    # plt.subplot(141)
    # plt.imshow(Dgeo, aspect='auto')
    # plt.title('Geometry init')
    # plt.subplot(142)
    # plt.imshow(Dsym, aspect='auto')
    # plt.title('Pyroom init')
    # plt.subplot(143)
    # plt.imshow(Dobs, aspect='auto')
    # plt.title('Peak picking')
    # plt.subplot(144)
    # plt.imshow(np.abs(Dobs - Dsym), aspect='auto')
    # plt.title('Diff')
    # plt.show()

    rmse = lambda x, y : np.sqrt(np.mean(np.abs(x - y)**2))


    print('# GEOM/SIGNAL MISSMATCH')
    Dgeo = Dgeo.copy()
    Dsgn = Dtoa.copy()
    Dcal = Dtoa.copy()

    print('## Before unfolding')
    print('### calib vs geo')
    me_i = np.max(np.abs(Dcal - Dgeo))
    mae_i = np.mean(np.abs(Dcal - Dgeo))
    rmse_i = rmse(Dcal, Dgeo)
    std_i = np.std(np.abs(Dcal - Dgeo))
    print('- input ME', me_i)
    print('- input MAE', mae_i)
    print('- input RMSE', rmse_i)
    print('- input std', std_i)

    print('### calib vs sig')
    me_i = np.max(np.abs(Dcal - Dsgn))
    mae_i = np.mean(np.abs(Dcal - Dsgn))
    rmse_i = rmse(Dcal, Dsgn)
    std_i = np.std(np.abs(Dcal - Dsgn))
    print('- input ME', me_i)
    print('- input MAE', mae_i)
    print('- input RMSE', rmse_i)
    print('- input std', std_i)



    print('UNFOLDING')
    if K == 0:
        X_est, A_est = nlls_mds_array(Dsgn, X, A)
        # X_est, A_est = nlls_mds(Dsgn, X, A)
    elif K == 1:
        X_est, A_est = nlls_mds_array_ceiling(Dsgn, De_c, X, A)
        # X_est, A_est = nlls_mds_ceiling(Dsgn, De_c, X, A)
    elif K == 2:
        X_est, A_est = nlls_mds_array_images(Dsgn, De_c, De_f, X, A)
        # X_est, A_est = nlls_mds_images(Dsgn, De_c, De_f, X, A)
    else:
        pass
    # mics_pos_est, srcs_pos_est = nlls_mds_array(Dtof, X, A)
    # mics_pos_est, srcs_pos_est = crcc_mds(D, init={'X': X, 'A': A}

    Dcal = edm(X_est, A_est)

    print('## AFTER unfolding')
    print('### calib vs geo')
    me_o = np.max(np.abs(Dgeo - Dcal))
    mae_o = np.mean(np.abs(Dgeo - Dcal))
    rmse_o = rmse(Dgeo, Dcal)
    std_o = np.std(np.abs(Dgeo - Dcal))
    print('- output ME', me_o)
    print('- output MAE', mae_o)
    print('- output RMSE', rmse_o)
    print('- output std', std_o)

    print('### calib vs sig')
    me_o = np.max(np.abs(Dsgn - Dcal))
    mae_o = np.mean(np.abs(Dsgn - Dcal))
    rmse_o = rmse(Dsgn, Dcal)
    std_o = np.std(np.abs(Dsgn - Dcal))
    print('- output ME', me_o)
    print('- output MAE', mae_o)
    print('- output RMSE', rmse_o)
    print('- output std', std_o)



    mics_pos_est = X_est
    srcs_pos_est = A_est

    # np.save(path_to_processed + 'mics_pos_est_nlls.npy', mics_pos_est)
    # np.save(path_to_processed + 'srcs_pos_est_nlls.npy', srcs_pos_est)
    # mics_pos_est = np.load(path_to_processed + 'mics_pos_est_nlls.npy')
    # srcs_pos_est = np.load(path_to_processed + 'srcs_pos_est_nlls.npy')


    # plt.imshow(rirs, extent=[0, I*J, 0, L], aspect='auto')
    for j in range(J):
        plt.axvline(j*30, color='C7')
    plt.axhline(y=L-recording_offset, label='Time of Emission')
    for k in range(K+1):
        wall = curr_reflectors[k]
        r = refl_order.index(wall)
        plt.scatter(np.arange(I*J)+0.5, L - recording_offset - toa_peak[r,:,:].T.flatten()*Fs, c='C1', label='Peak Picking')
        plt.scatter(np.arange(I*J)+0.5, L - recording_offset - toa_sym[r,:,:].T.flatten()*Fs, c='C2', label='Pyroom')

        if k == 0:
            A = A_est.copy()
        if k == 1:
            A = A_est.copy()
            A[2,:] = 2*Rz - A_est[2, :]

        if k == 2:
            A = A_est.copy()
            A[2, :] = - A_est[2, :]

        D = edm(X_est, A)
        new_tofs = D / speed_of_sound
        plt.scatter(np.arange(I*J)+0.5, L - recording_offset - new_tofs.T.flatten()*Fs, c='C3', marker='X', label='EDM k%d' % k)


    plt.tight_layout()
    plt.legend()
    plt.savefig('./reports/figures/rir_skyline_after_calibration_k-%d.pdf' % K)
    plt.show()

    return mics_pos_est, srcs_pos_est, mics_pos, srcs_pos, toa_sym, rirs


if __name__ == "__main__":

    datasets = constants['datasets']

    ## INITIALIZATION
    mics_pos_est = None
    srcs_pos_est = None
    dataset_id = '011111'

    # LOAD MANUAL ANNOTATION
    # path_to_manual_annotation = './data/interim/manual_annotation/20200422_21h03_gui_annotation.pkl'
    # path_to_manual_annotation = 'data/interim/manual_annotation/20200424_20h19_gui_annotation.pkl'
    # path_to_manual_annotation = 'data/interim/manual_annotation/20200428_12h27_gui_annotation.pkl'
    # path_to_manual_annotation = './data/processed/rirs_manual_annotation/20200505_12h38_gui_annotation.pkl'

    # annotation_file = '20200508_12h27_gui_annotation.pkl'
    # annotation_file = '20200508_13h33_gui_annotation.pkl'
    annotation_file = '20200508_16h29_gui_annotation.pkl'
    annotation_file = '20200508_18h34_gui_annotation.pkl' # good one, but the following problems:
    # src 0: bad south, west
    # src 1: bad north, east
    # src 2: bad south, west, east
    # annotation_file = '20200515_14h57_gui_annotation.pkl'  # starting fixing above problems
    path_to_manual_annotation = './data/processed/rirs_manual_annotation/' + annotation_file
    manual_note = load_from_pickle(path_to_manual_annotation)
    # we select only the reflection of order 0
    # namely one reflection coming from each wall
    toa_peak = manual_note['toa'][:7, :, :4, 0]

    ## K = 1: direct path estimation
    K = 0
    mics_pos_est, srcs_pos_est, mics_pos, srcs_pos, toa_sym, rirs \
        = iterative_calibration(dataset_id, mics_pos_est, srcs_pos_est, K, toa_peak)

    print(mics_pos.shape)
    print(mics_pos_est.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(mics_pos[0, :], mics_pos[1, :], mics_pos[2, :], marker='o', label='mics init')
    ax.scatter(srcs_pos[0, :], srcs_pos[1, :], srcs_pos[2, :], marker='o', label='srcs init')
    ax.scatter(mics_pos_est[0, :], mics_pos_est[1, :], mics_pos_est[2, :], marker='x', label='mics est')
    ax.scatter(srcs_pos_est[0, :], srcs_pos_est[1, :], srcs_pos_est[2, :], marker='x', label='srcs est')
    ax.set_xlim([0, Rx])
    ax.set_ylim([0, Ry])
    ax.set_zlim([0, Rz])
    plt.legend()
    plt.savefig('./reports/figures/cal_positioning3D_k0.pdf')
    plt.show()

    ## K = 1: echo 1 -- from the ceiling
    K = 1
    mics_pos_est, srcs_pos_est, mics_pos, srcs_pos, toa_sym, rirs \
        = iterative_calibration(dataset_id, mics_pos_est, srcs_pos_est, K, toa_peak)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(mics_pos[0, :], mics_pos[1, :], mics_pos[2, :], marker='o', label='mics init')
    ax.scatter(srcs_pos[0, :], srcs_pos[1, :], srcs_pos[2, :], marker='o', label='srcs init')
    ax.scatter(mics_pos_est[0, :], mics_pos_est[1, :], mics_pos_est[2, :], marker='x', label='mics est')
    ax.scatter(srcs_pos_est[0, :], srcs_pos_est[1, :], srcs_pos_est[2, :], marker='x', label='srcs est')
    ax.set_xlim([0, Rx])
    ax.set_ylim([0, Ry])
    ax.set_zlim([0, Rz])
    plt.legend()
    plt.savefig('./reports/figures/cal_positioning3D_k1.pdf')
    plt.show()


    calib_res = {
        'mics': mics_pos_est,
        'srcs': srcs_pos_est,
        'toa_pck' : toa_peak,
        'toa_sym' : toa_sym,
        'rirs' : rirs,
    }

    save_to_pickle('./data/processed/post2_calibration/calib_output_mics_srcs_pos.pkl', calib_res)

    # FINALLY, RIR SKYLINE WITH ALL THE ECHOES
    curr_reflectors = constants['refl_order_calibr'][:7]
    refl_order = constants['refl_order_pyroom']

    L, I, J = rirs.shape
    rirs_skyline = rirs.transpose([0, 2, 1]).reshape([L, I*J])
    plt.imshow(rirs_skyline, extent=[0, I*J, 0, L], aspect='auto')

    # plot srcs boundaries
    for j in range(J):
            plt.axvline(j*30, color='C7')

    # plot time of emission
    # plt.axhline(y=L-recording_offset, label='Time of Emission')

    for k in range(7):
        print(curr_reflectors)
        wall = curr_reflectors[k]
        r = refl_order.index(wall)
        # plot peak annotation
        plt.scatter(np.arange(I*J)+0.5, L - recording_offset - toa_peak[r, :, :].T.flatten()*Fs,
                    c='C%d' % (k+2), marker='x', label='%s Picking' % wall)
        # plot simulated peak
        plt.scatter(np.arange(I*J)+0.5, L - recording_offset - toa_sym[r, :, :].T.flatten()*Fs,
                    marker='o', facecolors='none', edgecolors='C%d' % (k+2), label='%s Pyroom' % wall)

    plt.tight_layout()
    plt.legend(ncol=7)
    # plt.title('RIR SKYLINE K = %d' % K)
    plt.savefig('./reports/figures/rir_skyline_final.pdf')
    plt.show()

    print(np.abs(toa_peak[:,0,0] - toa_sym[:,0,0]))

    ## save here for the GUI later
    new_manual_note = manual_note.copy()
    new_manual_note['toa'][:7, :, :4, 0] = toa_sym[:7, :, :4]
    save_to_pickle('./data/processed/post2_calibration/calib_output_toa_pck.pkl', new_manual_note)
    new_manual_note['toa'][:7, :, :4, 0] = toa_peak[:7, :, :4]
    save_to_pickle('./data/processed/post2_calibration/calib_output_toa_sym.pkl', new_manual_note)


    # ## K = 2: echo 1,2 -- from the ceiling and the floor
    # K = 2
    # mics_pos_est, srcs_pos_est, mics_pos, srcs_pos, toa_sym \
    #     = iterative_calibration(dataset_id, mics_pos_est, srcs_pos_est, K, toa_peak)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(mics_pos[0, :], mics_pos[1, :],
    #            mics_pos[2, :], marker='o', label='mics init')
    # ax.scatter(srcs_pos[0, :], srcs_pos[1, :],
    #            srcs_pos[2, :], marker='o', label='srcs init')
    # ax.scatter(mics_pos_est[0, :], mics_pos_est[1, :],
    #            mics_pos_est[2, :], marker='x', label='mics est')
    # ax.scatter(srcs_pos_est[0, :], srcs_pos_est[1, :],
    #            srcs_pos_est[2, :], marker='x', label='srcs est')
    # ax.set_xlim([0, Rx])
    # ax.set_ylim([0, Ry])
    # ax.set_zlim([0, Rz])
    # plt.legend()
    # plt.savefig('./reports/figures/cal_positioning3D.pdf')
    # plt.show()


    # # save current refine TOAs
    # print(toa_sym)
    # manual_note['toa'][:7, :, :, 0] = toa_sym
    # path_to_output_toa = './data/interim/toa_after_calibration.pkl'
    # save_to_pickle(path_to_output_toa, manual_note)

    # ## K = 3: echo 1,2,3 -- from the ceiling and the floor, west
    # K = 3
    # mics_pos_est, srcs_pos_est, mics_pos, srcs_pos, toa_sym \
    #     = iterative_calibration(dataset_id, mics_pos_est, srcs_pos_est, K, toa_peak)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(mics_pos[0, :], mics_pos[1, :],
    #            mics_pos[2, :], marker='o', label='mics init')
    # ax.scatter(srcs_pos[0, :], srcs_pos[1, :],
    #            srcs_pos[2, :], marker='o', label='srcs init')
    # ax.scatter(mics_pos_est[0, :], mics_pos_est[1, :],
    #            mics_pos_est[2, :], marker='x', label='mics est')
    # ax.scatter(srcs_pos_est[0, :], srcs_pos_est[1, :],
    #            srcs_pos_est[2, :], marker='x', label='srcs est')
    # ax.set_xlim([0, Rx])
    # ax.set_ylim([0, Ry])
    # ax.set_zlim([0, Rz])
    # plt.legend()
    # plt.savefig('./reports/figures/cal_positioning3D.pdf')
    # plt.show()

    pass



 # # Blueprint 2D xz plane
# room_size = [5.543, 5.675, 2.353]
# plt.figure(figsize=(16, 9))
# plt.gca().add_patch(
#     plt.Rectangle((0, 0),
#                 room_size[0], room_size[2], fill=False,
#                 edgecolor='g', linewidth=1)
# )
# plt.scatter(mics_pos[0, :], mics_pos[2, :], marker='X', label='mic init')
# plt.scatter(mics_pos_est[0, :], mics_pos_est[2, :], marker='X', label='mic est')
# plt.scatter(srcs_pos[0, :], srcs_pos[2, :], marker='v', label='src init')
# plt.scatter(srcs_pos_est[0, :], srcs_pos_est[2, :], marker='v', label='src est')
# for i in range(I):
#     if i % 5 == 0:
#         bar = np.mean(mics_pos[:, 5*i//5:5*(i//5+1)], axis=1)
#         plt.text(bar[0], bar[2], '$arr_%d$' % (i//5 + 1), fontdict={'fontsize': 8})
#         bar = np.mean(mics_pos_est[:, 5*i//5:5*(i//5+1)], axis=1)
#         plt.text(bar[0], bar[2], '$arr_%d$' %(i//5 + 1), fontdict={'fontsize': 8})
# for j in range(J):
#     bar = srcs_pos[:, j]
#     if j < 6:
#         plt.text(bar[0], bar[2], '$dir_%d$' % (j+1), fontdict={'fontsize': 8})
#     else:
#         plt.text(bar[0], bar[2], '$omn_%d$' % (j+1), fontdict={'fontsize': 8})
#     bar = srcs_pos_est[:, j]
#     if j < 6:
#         plt.text(bar[0], bar[2], '$dir_%d$' % (j+1), fontdict={'fontsize': 8})
#     else:
#         plt.text(bar[0], bar[2], '$omn_%d$' % (j+1), fontdict={'fontsize': 8})
# plt.legend()
# plt.title('Projection: xz')
# plt.savefig('./reports/figures/cal_positioning2D_xz.pdf')
# plt.show()

# # Blueprint 2D xy plane
# room_size = [5.543, 5.675, 2.353]
# plt.figure(figsize=(16, 9))
# plt.gca().add_patch(
#     plt.Rectangle((0, 0),
#                 room_size[0], room_size[1], fill=False,
#                 edgecolor='g', linewidth=1)
# )

# plt.scatter(mics_pos[0, :], mics_pos[1, :], marker='X', label='mic init')
# plt.scatter(mics_pos_est[0, :], mics_pos_est[1, :], marker='X', label='mic est')
# plt.scatter(srcs_pos[0, :], srcs_pos[1, :], marker='v', label='src init')
# plt.scatter(srcs_pos_est[0, :], srcs_pos_est[1, :], marker='v', label='src est')
# for i in range(I):
#     if i % 5 == 0:
#         bar = np.mean(mics_pos[:, 5*i//5:5*(i//5+1)], axis=1)
#         plt.text(bar[0], bar[1], '$arr_%d$' % (i//5 + 1), fontdict={'fontsize': 8})
#         bar = np.mean(mics_pos_est[:, 5*i//5:5*(i//5+1)], axis=1)
#         plt.text(bar[0], bar[1], '$arr_%d$' %(i//5 + 1), fontdict={'fontsize': 8})
# for j in range(J):
#     bar = srcs_pos[:, j]
#     if j < 6:
#         plt.text(bar[0], bar[1], '$dir_%d$' % (j+1), fontdict={'fontsize': 8})
#     else:
#         plt.text(bar[0], bar[1], '$omn_%d$' % (j+1), fontdict={'fontsize': 8})
#     bar = srcs_pos_est[:, j]
#     if j < 6:
#         plt.text(bar[0], bar[1], '$dir_%d$' % (j+1), fontdict={'fontsize': 8})
#     else:
#         plt.text(bar[0], bar[1], '$omn_%d$' % (j+1), fontdict={'fontsize': 8})
# plt.legend()
# plt.title('Projection: xy')
# plt.savefig('./reports/figures/cal_positioning2D_xy.pdf')
# plt.show()
