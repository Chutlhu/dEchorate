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

from src.utils.file_utils import save_to_matlab
from src.utils.dsp_utils import envelope
from src import constants

from risotto import deconvolution as deconv

Fs = constants['Fs'] # Sampling frequency
recording_offset = constants['recording_offset']
T = 24     # temperature
speed_of_sound = 331.3 + 0.606 * T # speed of sound
L = int(0.5*Fs) # max length of the filter


def compute_distances_from_rirs(path_to_dataset_rir, dataset):
    f_rir = h5py.File(path_to_dataset_rir, 'r')

    all_src_ids = np.unique(dataset['src_id'])
    all_src_ids = all_src_ids[~np.isnan(all_src_ids)]
    all_mic_ids = np.unique(dataset['mic_id'])
    all_mic_ids = all_mic_ids[~np.isnan(all_mic_ids)]

    I = len(all_mic_ids)
    J = len(all_src_ids)

    mics_pos = np.zeros([3, I])
    srcs_pos = np.zeros([3, J])

    tofs_simulation = np.zeros([I, J])
    manual_toa = np.zeros([I, J])
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

            x = np.arange(0, len(rir))
            y = rir
            f = intp.interp1d(x, y, kind='cubic')
            xnew = np.arange(0, len(rir)-1, 0.1)
            ynew = f(xnew)
            ynew = np.abs(ynew / np.max(np.abs(ynew)))

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
            # peaks, _ = sg.find_peaks(rir_abs, height=0.2, distance=50, width=2, prominence=0.6)
            # dp_peak = np.min(peaks)

            peaks, _ = sg.find_peaks(ynew, height=0.2, distance=50, width=2, prominence=0.6)
            dp_peak = np.min(peaks)

            # # floor
            # d_min_floor_ceiling = np.abs(tof_geom_ceiling - tof_geom_floor)*Fs
            # peaks, _ = sg.find_peaks(rir_abs[thr1:thr2], height=0.2, distance=50, width=2, prominence=0.2)
            # peaks = peaks + thr1

            # ## MANUAL ANNOTATION
            # print("mic %d/%d\tsrc %d/%d" % (i, I, j, J))
            # plt.plot(rir_abs)
            # plt.plot(rir_abs**2, alpha=0.5)
            # plt.plot(xnew, ynew)
            # plt.scatter(xnew[dp_peak], ynew[dp_peak])
            # plt.axvline(tof_geom*Fs + recording_offset)
            # plt.xlim(xnew[dp_peak] - 20, xnew[dp_peak] + 20)
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

            # extract the time of arrival from the RIR
            direct_path_positions[ij] = xnew[dp_peak]

            toa = direct_path_positions[ij]/Fs

            tofs_simulation[i, j] = tof_geom
            toes_rir[i, j] = recording_offset/Fs
            toas_rir[i, j] = toa
            tofs_rir[i, j] = toa - recording_offset/Fs

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

            ij += 1

    return rirs, recording_offset, tofs_simulation, tofs_rir, mics_pos, srcs_pos


if __name__ == "__main__":
    dataset_dir = './data/dECHORATE/'
    path_to_processed = './data/processed/'

    session_id = '000000' # '010000'

    path_to_dataset_rir = path_to_processed + '%s_rir_data.hdf5' % session_id

    path_to_database = dataset_dir + 'annotations/dECHORATE_database.csv'
    dataset = pd.read_csv(path_to_database)
    # select dataset with entries according to session_id
    f, c, w, e, n, s = [int(i) for i in list(session_id)]
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

    ## COMPUTE DIRECT PATH POSITIONS
    rirs, recording_offset, tofs_simulation, tofs_rir, mics_pos, srcs_pos \
        = compute_distances_from_rirs(path_to_dataset_rir, dataset)

    # some hard coded variables
    tofs_rir[20, 0] = (4715 - 4444)/Fs
    tofs_rir[21, 0] = (4714 - 4444)/Fs
    tofs_rir[22, 0] = (4711 - 4444)/Fs


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
    plt.close()

    # save_to_matlab(path_to_processed + 'src_mic_dist.mat', tofs_rir)

    # ## MULTIDIMENSIONAL SCALING
    # select sub set of microphones and sources

    # # nonlinear least square problem with good initialization
    X = mics_pos[:, :I]
    A = srcs_pos[:, 0:4]
    print(A.shape)

    # D = tofs_simulation * speed_of_sound
    Dedm = edm(X, A) ** (.5)
    Dtof = tofs_rir[:I, 0:4] * speed_of_sound
    Dgeo = tofs_simulation[:I, 0:4] * speed_of_sound
    assert np.allclose(Dedm, Dgeo)
    mics_pos_est, srcs_pos_est = nlls_mds(Dtof, init={'X': X, 'A': A})
    # mics_pos_est, srcs_pos_est = crcc_mds(D, init={'X': X, 'A': A})

    np.save(path_to_processed + 'mics_pos_est_nlls.npy', mics_pos_est)
    np.save(path_to_processed + 'srcs_pos_est_nlls.npy', srcs_pos_est)
    mics_pos_est = np.load(path_to_processed + 'mics_pos_est_nlls.npy')
    srcs_pos_est = np.load(path_to_processed + 'srcs_pos_est_nlls.npy')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[0, :], X[1, :], X[2, :], marker='o', label='mics init')
    ax.scatter(A[0, :], A[1, :], A[2, :], marker='o', label='srcs init')
    ax.scatter(mics_pos_est[0, :], mics_pos_est[1, :], mics_pos_est[2, :], marker='x', label='mics est')
    ax.scatter(srcs_pos_est[0, :], srcs_pos_est[1, :], srcs_pos_est[2, :], marker='x', label='srcs est')
    plt.legend()
    plt.savefig('./reports/figures/cal_positioning3D.pdf')
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
    plt.close()

    # Blueprint 2D xz plane
    room_size = [5.543, 5.675, 2.353]
    plt.figure(figsize=(16, 9))
    plt.gca().add_patch(
        plt.Rectangle((0, 0),
                    room_size[0], room_size[2], fill=False,
                    edgecolor='g', linewidth=1)
    )
    plt.scatter(mics_pos[0, :], mics_pos[2, :], marker='X', label='mic init')
    plt.scatter(mics_pos_est[0, :], mics_pos_est[2, :], marker='X', label='mic est')
    plt.scatter(srcs_pos[0, :], srcs_pos[2, :], marker='v', label='src init')
    plt.scatter(srcs_pos_est[0, :], srcs_pos_est[2, :], marker='v', label='src est')
    for i in range(I):
        if i % 5 == 0:
            bar = np.mean(mics_pos[:, 5*i//5:5*(i//5+1)], axis=1)
            plt.text(bar[0], bar[2], '$arr_%d$' % (i//5 + 1), fontdict={'fontsize': 8})
            bar = np.mean(mics_pos_est[:, 5*i//5:5*(i//5+1)], axis=1)
            plt.text(bar[0], bar[2], '$arr_%d$' %(i//5 + 1), fontdict={'fontsize': 8})
    for j in range(J):
        bar = srcs_pos[:, j]
        if j < 6:
            plt.text(bar[0], bar[2], '$dir_%d$' % (j+1), fontdict={'fontsize': 8})
        else:
            plt.text(bar[0], bar[2], '$omn_%d$' % (j+1), fontdict={'fontsize': 8})
        bar = srcs_pos_est[:, j]
        if j < 6:
            plt.text(bar[0], bar[2], '$dir_%d$' % (j+1), fontdict={'fontsize': 8})
        else:
            plt.text(bar[0], bar[2], '$omn_%d$' % (j+1), fontdict={'fontsize': 8})
    plt.legend()
    plt.title('Projection: xz')
    plt.savefig('./reports/figures/cal_positioning2D_xz.pdf')
    plt.show()

    # Blueprint 2D xy plane
    room_size = [5.543, 5.675, 2.353]
    plt.figure(figsize=(16, 9))
    plt.gca().add_patch(
        plt.Rectangle((0, 0),
                    room_size[0], room_size[1], fill=False,
                    edgecolor='g', linewidth=1)
    )

    plt.scatter(mics_pos[0, :], mics_pos[1, :], marker='X', label='mic init')
    plt.scatter(mics_pos_est[0, :], mics_pos_est[1, :], marker='X', label='mic est')
    plt.scatter(srcs_pos[0, :], srcs_pos[1, :], marker='v', label='src init')
    plt.scatter(srcs_pos_est[0, :], srcs_pos_est[1, :], marker='v', label='src est')
    for i in range(I):
        if i % 5 == 0:
            bar = np.mean(mics_pos[:, 5*i//5:5*(i//5+1)], axis=1)
            plt.text(bar[0], bar[1], '$arr_%d$' % (i//5 + 1), fontdict={'fontsize': 8})
            bar = np.mean(mics_pos_est[:, 5*i//5:5*(i//5+1)], axis=1)
            plt.text(bar[0], bar[1], '$arr_%d$' %(i//5 + 1), fontdict={'fontsize': 8})
    for j in range(J):
        bar = srcs_pos[:, j]
        if j < 6:
            plt.text(bar[0], bar[1], '$dir_%d$' % (j+1), fontdict={'fontsize': 8})
        else:
            plt.text(bar[0], bar[1], '$omn_%d$' % (j+1), fontdict={'fontsize': 8})
        bar = srcs_pos_est[:, j]
        if j < 6:
            plt.text(bar[0], bar[1], '$dir_%d$' % (j+1), fontdict={'fontsize': 8})
        else:
            plt.text(bar[0], bar[1], '$omn_%d$' % (j+1), fontdict={'fontsize': 8})
    plt.legend()
    plt.title('Projection: xy')
    plt.savefig('./reports/figures/cal_positioning2D_xy.pdf')
    plt.show()

    pass
