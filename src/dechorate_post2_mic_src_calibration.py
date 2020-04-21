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

from src import constants

from src.dataset import DechorateDataset, SyntheticDataset
from src.calibration_and_mds import nlls_mds, nlls_mds_array, nlls_mds_ceiling

from src.utils.file_utils import save_to_matlab, load_from_pickle
from src.utils.dsp_utils import envelope, normalize
from src.utils.mds_utils import edm

from risotto import deconvolution as deconv

Fs = constants['Fs'] # Sampling frequency
recording_offset = constants['recording_offset']
Rx, Ry, Rz = constants['room_size']
speed_of_sound = constants['speed_of_sound']  # speed of sound

dataset_dir = './data/dECHORATE/'
path_to_processed = './data/processed/'


def load_rirs(path_to_dataset_rir, dataset, K,
              dataset_id, mics_pos=None, srcs_pos=None):

    f_rir = h5py.File(path_to_dataset_rir, 'r')

    all_src_ids = np.unique(dataset['src_id'])
    all_src_ids = all_src_ids[~np.isnan(all_src_ids)]
    all_mic_ids = np.unique(dataset['mic_id'])
    all_mic_ids = all_mic_ids[~np.isnan(all_mic_ids)]

    I = len(all_mic_ids)
    J = len(all_src_ids)

    # if mics_pos is None:
    # if srcs_pos is None:
    mics_pos = np.zeros([3, I])
    srcs_pos = np.zeros([3, J])


    toa_sym = np.zeros([7, I, J])
    toa_peak = np.zeros([7, I, J])

    amp_sym = np.zeros([7, I, J])
    ord_sym = np.zeros([7, I, J])
    wal_sym = np.chararray([7, I, J])

    L = int(0.5*Fs) # max length of the filter
    rirs = np.zeros([L, I*J])

    ij = 0
    for j in tqdm(range(J)):
        for i in range(I):

            # find recording correspondent to mic i and src j
            entry = dataset.loc[(dataset['src_id'] == j+1) & (dataset['mic_id'] == i+1)]

            assert len(entry) == 1

            wavefile = entry['filename'].values[0]

            rir = f_rir['rir/%s/%d' % (wavefile, i)][()].squeeze()

            rir = rir/np.max(np.abs(rir))
            rirs[:, ij] = np.abs(rir)

            x = np.arange(0, len(rir))
            y = rir
            f = intp.interp1d(x, y, kind='cubic')
            xnew = np.arange(0, len(rir)-1, 0.1)
            ynew = f(xnew)
            ynew = np.abs(ynew / np.max(np.abs(ynew)))

            # compute the theoretical distance
            if np.allclose(mics_pos[:, i], 0):
                mic_pos = [entry['mic_pos_x'].values, entry['mic_pos_y'].values, entry['mic_pos_z'].values]
                mics_pos[:, i] = np.array(mic_pos).squeeze()

            if np.allclose(srcs_pos[:, j], 0):
                src_pos = [entry['src_pos_x'].values, entry['src_pos_y'].values, entry['src_pos_z'].values]
                srcs_pos[:, j] = np.array(src_pos).squeeze()

            synth_dset = SyntheticDataset()
            synth_dset.set_room_size(constants['room_size'])
            synth_dset.set_dataset(dataset_id, absb=1, refl=0)
            synth_dset.set_c(speed_of_sound)
            synth_dset.set_k_order(1)
            synth_dset.set_k_reflc(7)
            synth_dset.set_mic(mics_pos[0, i], mics_pos[1, i], mics_pos[2, i])
            synth_dset.set_src(srcs_pos[0, j], srcs_pos[1, j], srcs_pos[2, j])
            amp, tau, wall, order = synth_dset.get_note()

            toa_sym[:, i, j] = tau
            amp_sym[:, i, j] = amp
            wal_sym[:, i, j] = wall
            ord_sym[:, i, j] = order

            ij += 1

            # idx_walls = np.nonzero(amp_sym[:, i, j])[0]
            # for c, k in enumerate(idx_walls):
            #     t = (recording_offset + tau[k]*Fs)
            #     p = np.argmin(np.abs(xnew - t))
            #     print(p)
            #     if k == 0:
            #         idx = [p - 200, p + 200]
            #         # direct path peak from the interpolated
            #         tmp = ynew[idx[0]:idx[1]]
            #         peaks, _ = sg.find_peaks(tmp, height=0.2, distance=50, width=2, prominence=0.6)
            #         # peaks, _ = sg.find_peaks(tmp, height=0.2, distance=50, width=2, prominence=0.6)

            #         plt.plot(ynew)
            #         plt.scatter(peaks + idx[0], ynew[peaks + idx[0]])
            #         plt.show()

            #     else:
            #         idx = [p - 300, p + 300]
            #         # direct path peak from the interpolated
            #         tmp = ynew[idx[0]:idx[1]]
            #         peaks, _ = sg.find_peaks(tmp)


            #     peak = idx[0] + np.min(peaks[np.argmax(tmp[peaks])])
            #     toa_peak[c, i, j] = (xnew[peak] - recording_offset)/Fs
            #     toa_sym[c, i, j] = tau[k]

            #     plt.figure(figsize=(16, 9))
            #     print(wal_sym[:, i, j])
            #     print(toa_sym[:, i, j])
            #     print(amp_sym[:, i, j])
            #     plt.title('mic %d (%d), src %d' % (i, i+33, j))
            #     plt.plot(ynew, label='recorded rir')
            #     p0 = (recording_offset + toa_sym[0, i, j]*Fs)*10
            #     p1 = (recording_offset + toa_sym[1, i, j]*Fs)*10
            #     plt.axvline(p0, color='red', ls='--', label='dp sym')
            #     plt.axvline(p1, color='green', ls='--', label='e1 sym')

            #     p0 = (recording_offset + toa_peak[0, i, j]*Fs)*10
            #     p1 = (recording_offset + toa_peak[1, i, j]*Fs)*10
            #     plt.axvline(p0, color='red', label='dp peak')
            #     plt.axvline(p1, color='green', label='e1 peak')
            #     plt.legend()
            #     plt.show()

            #     #     for k in range(7):
            #     #         pk = (recording_offset + tau[k]*Fs)*10
            #     #         plt.axvline(pk, color='black', ls='--', alpha=0.3)
            #     #         plt.annotate(r'$\tau_{%d}^{%s}$' % (pk, wal_sym[k, i, j].decode()), [pk, 0.5])

            #     #     plt.xlim([p0-1000, p1+4000])
            #     #     plt.show()

            #     assert len(peaks) > 0

            # plt.axvline(toa_peak[:, i, j], color='C5', label='dp')
            # plt.plot(np.abs(normalize(rir)), label='original')
            # plt.legend()
            # plt.show()

            # CEILING-tuned PEAK PICKING
            # peak_ceiling = 4444+tof1_geom*Fs

            # plt.scatter(peaks, rir_deconv[peaks], marker='x', label='peaks')
            # plt.scatter(peak_ceiling, rir_deconv[peak_ceiling], marker='d', color='C5', label='ceiling')
            # plt.show()

            # plt.figure(figsize=[12,9])
            # plt.title("mic %d/%d  src %d/%d" % (i+33, I, j, J))
            # plt.axvline(4444+tof_geom*Fs, color='C4', ls='--', label='dp_geom')
            # plt.axvline(4444+tof1_geom*Fs, color='C4', ls='--', label='e1_geom')
            # plt.plot(np.abs(normalize(rir)), label='original')
            # plt.plot(np.abs(normalize(rir_deconv)), label='original', alpha=0.8)
            # plt.axvline(xnew[dp_peak], color='C5', label='dp')
            # plt.axvline(xnew[e1_peak], color='C5', label='e1')
            # plt.xlim([xnew[dp_peak]-50, xnew[e1_peak]+100])
            # plt.legend()
            # plt.show()



            ## MANUAL ANNOTATION
            # print("mic %d/%d\tsrc %d/%d" % (i, I, j, J))
            # plt.plot(rir_abs)
            # # plt.plot(rir_abs**2, alpha=0.5)
            # plt.plot(xnew, ynew)
            # plt.plot(xnew_deconv, ynew_deconv)
            # plt.scatter(xnew[dp_peak], ynew[dp_peak])
            # plt.scatter(xnew_deconv[peaks], ynew_deconv[peaks])
            # plt.axvline(tof_geom*Fs + recording_offset)
            # # plt.xlim(xnew[dp_peak] - 20, xnew[dp_peak] + 20)
            # plt.show()

            # txt = input()
            # print(txt)
            # if txt == '':
            #     manual_toa[i, j] = xnew[dp_peak]/Fs - recording_offset/Fs
            # else:
            #     manual_toa[i, j] = float(txt)/Fs - recording_offset/Fs

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
            # plt.title('Source %d, array %d, microphone %d' % (j+1, i//5 + 1, i % 5+1))
            # plt.plot(rir_abs)
            # plt.plot(peaks, rir_abs[peaks], "x")
            # plt.plot(dp_peak, rir_abs[dp_peak], "o")
            # plt.axvline(x=recording_offset, color='C0', label='offset')
            # plt.axvline(x=tof_geom*Fs+recording_offset, color='C0', label='direct')
            # # plt.axvline(x=tof_geom_ceiling*Fs+recording_offset, color='C1', label='ceiling')
            # plt.axvline(x=tof_geom_floor*Fs+recording_offset, color='C2', label='floor')
            # plt.legend()
            # plt.xlim([recording_offset-100, recording_offset+800])
            # plt.show()

            # np.savetxt('./data/processed/rirs_manual_annotation/from_post2.csv', manual_toa)

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

    return rirs, toa_sym, mics_pos, srcs_pos


def iterative_calibration(dataset_id, mics_pos, src_pos, K):

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
    rirs, toa_sym, mics_pos, srcs_pos = load_rirs(path_to_dataset_rir, dataset, K, dataset_id, mics_pos, src_pos)
    print(toa_sym.shape)

    # LOAD MANUAL ANNOTATION
    path_to_manual_annotation = './data/processed/rirs_manual_annotation/20200410_22h47_gui_rir_annotation.pkl'
    manual_note = load_from_pickle(path_to_manual_annotation)
    toa_peak = manual_note['toa'][:7, :, :, 0]

    assert toa_peak.shape == toa_sym.shape
    assert toa_peak.shape[1] == mics_pos.shape[1]
    assert toa_peak.shape[2] == srcs_pos.shape[1]

    L, IJ = rirs.shape
    D, I = mics_pos.shape
    D, J = srcs_pos.shape

    plt.imshow(rirs, extent=[0, I*J, 0, L], aspect='auto')
    for j in range(J):
            plt.axvline(j*30, color='C7')
    plt.axhline(y=L-recording_offset, label='Time of Emission')
    for k in range(K+1):
        wall = curr_reflectors[k]
        r = refl_order.index(wall)

        plt.scatter(np.arange(I*J)+0.5, L - recording_offset - toa_peak[r,:,:].T.flatten()*Fs, c='C1', label='Peak Picking')
        plt.scatter(np.arange(I*J)+0.5, L - recording_offset - toa_sym[r,:,:].T.flatten()*Fs, c='C2', label='Pyroom')
    plt.tight_layout()
    plt.legend()
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
        Dobs = toa_peak[r, :I, :J] * speed_of_sound
        Dsym = toa_sym[r, :I, :J] * speed_of_sound
    if K == 1:
        Dobs = toa_peak[0, :I, :J] * speed_of_sound
        Dsym = toa_sym[0, :I, :J] * speed_of_sound
        wall = curr_reflectors[k]
        r = refl_order.index(wall)
        De1 = toa_peak[r, :I, :J] * speed_of_sound

    assert np.allclose(Dgeo, Dsym)

    plt.subplot(141)
    plt.imshow(Dgeo, aspect='auto')
    plt.title('Geometry init')
    plt.subplot(142)
    plt.imshow(Dsym, aspect='auto')
    plt.title('Pyroom init')
    plt.subplot(143)
    plt.imshow(Dobs, aspect='auto')
    plt.title('Peak picking')
    plt.subplot(144)
    plt.imshow(np.abs(Dobs - Dsym), aspect='auto')
    plt.title('Diff')
    plt.show()

    rmse = lambda x, y : np.sqrt(np.mean(np.abs(x - y)**2))

    print('Initial margin', np.linalg.norm(Dsym - Dobs))
    print('Initial rmse',   rmse(Dsym, Dobs))
    if K == 0:
        X_est, A_est = nlls_mds(Dobs, X, A)
    elif K == 1:
        X_est, A_est = nlls_mds_ceiling(Dobs, De1, X, A)
    else:
        pass
    # mics_pos_est, srcs_pos_est = nlls_mds_array(Dtof, X, A)
    # mics_pos_est, srcs_pos_est = crcc_mds(D, init={'X': X, 'A': A})
    Dgeo_est = edm(X_est, A_est)
    print('After unfolding nlls', np.linalg.norm(Dobs - Dgeo_est))

    me = np.max(np.abs(Dobs - Dgeo_est))
    mae =  np.mean(np.abs(Dobs - Dgeo_est))
    rmse = rmse(Dobs, Dgeo_est)
    std = np.std(np.abs(Dobs - Dgeo_est))

    print('ME', me)
    print('MAE', mae)
    print('RMSE', rmse)
    print('std', std)

    mics_pos_est = X_est
    srcs_pos_est = A_est

    np.save(path_to_processed + 'mics_pos_est_nlls.npy', mics_pos_est)
    np.save(path_to_processed + 'srcs_pos_est_nlls.npy', srcs_pos_est)
    mics_pos_est = np.load(path_to_processed + 'mics_pos_est_nlls.npy')
    srcs_pos_est = np.load(path_to_processed + 'srcs_pos_est_nlls.npy')

    new_tofs = Dgeo_est / speed_of_sound

    for j in range(J):
        plt.axvline(j*30, color='C7')
    plt.axhline(y=L-recording_offset, label='Time of Emission')
    for k in range(K):
        plt.scatter(np.arange(I*J)+0.5, L - recording_offset - toa_peak[k,:,:].T.flatten()*Fs, c='C1', label='Peak Picking')
        plt.scatter(np.arange(I*J)+0.5, L - recording_offset - toa_sym[k,:,:].T.flatten()*Fs, c='C2', label='Pyroom')
    plt.scatter(np.arange(I*J)+0.5, L - recording_offset - new_tofs.T.flatten()*Fs, c='C3', marker='X', label='After EDM')
    plt.tight_layout()
    plt.legend()
    plt.savefig('./reports/figures/rir_skyline_after_calibration.pdf')
    plt.show()

    return mics_pos_est, srcs_pos_est, mics_pos, srcs_pos


if __name__ == "__main__":

    datasets = constants['datasets']

    ## INITIALIZATION
    mics_pos = None
    srcs_pos = None
    dataset_id = '011111'

    # ## K = 1: direct path estimation
    # K = 0
    # mics_pos_est, srcs_pos_est, mics_pos, srcs_pos \
    #     = iterative_calibration(dataset_id, mics_pos, srcs_pos, K)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(mics_pos[0, :], mics_pos[1, :], mics_pos[2, :], marker='o', label='mics init')
    # ax.scatter(srcs_pos[0, :], srcs_pos[1, :], srcs_pos[2, :], marker='o', label='srcs init')
    # ax.scatter(mics_pos_est[0, :], mics_pos_est[1, :], mics_pos_est[2, :], marker='x', label='mics est')
    # ax.scatter(srcs_pos_est[0, :], srcs_pos_est[1, :], srcs_pos_est[2, :], marker='x', label='srcs est')
    # ax.set_xlim([0, Rx])
    # ax.set_ylim([0, Ry])
    # ax.set_zlim([0, Rz])
    # plt.legend()
    # plt.savefig('./reports/figures/cal_positioning3D.pdf')
    # plt.show()

    ## K = 1: echo 1 -- from the ceiling
    K = 1
    # mics_pos = mics_pos_est
    # srcs_pos = srcs_pos_est
    mics_pos_est, srcs_pos_est, mics_pos, srcs_pos \
        = iterative_calibration(dataset_id, mics_pos, srcs_pos, K)

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
    plt.savefig('./reports/figures/cal_positioning3D.pdf')
    plt.show()

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
