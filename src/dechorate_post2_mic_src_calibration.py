import h5py
import numpy as np
import pandas as pd
import scipy.signal as sg

import matplotlib.pyplot as plt

from tqdm import tqdm

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

    tofs_simulation = np.zeros([I, J])
    toes_rir = np.zeros([I, J])
    tofs_rir = np.zeros([I, J])
    toas_rir = np.zeros([I, J])

    rirs = np.zeros([L, I*J])
    direct_path_positions = np.zeros([I*J])
    prev = 6444
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
            mic_pos = np.array(mic_pos)

            src_pos = [entry['src_pos_x'].values, entry['src_pos_y'].values, entry['src_pos_z'].values]
            src_pos = np.array(src_pos)

            d = np.linalg.norm(mic_pos - src_pos)
            tof_geom = d / speed_of_sound

            # extract the time of arrival from the RIR
            direct_path_positions[ij] = np.min(peaks)
            recording_offset = f_rir['delay/%s/%d' % (wavefile, i)][()]
            if not recording_offset == prev:
                print(recording_offset)
                recording_offset = prev
            prev = recording_offset

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
        plt.axvline(j*30)
    plt.axhline(y=L-recording_offset)
    plt.scatter(np.arange(I*J)+0.5, L-direct_path_positions, c='C1')
    plt.scatter(np.arange(I*J)+0.5,
                L - recording_offset - tofs_simulation.T.flatten()*Fs, c='C2')
    plt.tight_layout()
    plt.savefig('./reports/figures/rir_skyline')
    plt.show()

    return src_mic_dist


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
    src_mic_dist = compute_distances_from_rirs(
        path_to_anechoic_dataset_rir, anechoic_dataset_chirp)

    plt.imshow(src_mic_dist)
    plt.show()
    pass
