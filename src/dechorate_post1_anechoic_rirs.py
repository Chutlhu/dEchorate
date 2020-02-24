import os
import h5py
import numpy as np
import scipy.signal as sg
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt

from zipfile import ZipFile
from tqdm import tqdm

from src.stimulus import ProbeSignal
from src.utils.dsp_utils import *

# opening the zip file in READ mode
def get_zipfile_list(path_to_file):
    with ZipFile(path_to_file, 'r') as zip:
        return zip.namelist()


def get_zipped_file(filename, path_to_zipfile, path_to_output):
    with ZipFile(path_to_zipfile, 'r') as zip:
        return zip.extract(filename, path_to_output)


def silence_and_chirps_names(path_to_note):
    dataset_note = pd.read_csv(path_to_note)
    # select the annechoic recordings and only chirps
    anechoic_chirps = dataset_note.loc[
        (dataset_note['floor'] == 0 )
        & (dataset_note['ceiling'] == 0)
        & (dataset_note['west'] == 0)
        & (dataset_note['east'] == 0)
        & (dataset_note['north'] == 0)
        & (dataset_note['south'] == 0)
        & (dataset_note['signal'] == 'chirp')
    ]
    anechoic_silence = dataset_note.loc[
        (dataset_note['floor'] == 0)
        & (dataset_note['ceiling'] == 0)
        & (dataset_note['west'] == 0)
        & (dataset_note['east'] == 0)
        & (dataset_note['north'] == 0)
        & (dataset_note['south'] == 0)
        & (dataset_note['signal'] == 'silence')
    ]
    # check that there are not detrimental artifacts
    assert np.all(anechoic_chirps['artifacts'] < 2)
    assert np.all(anechoic_silence['artifacts'] < 2)
    # get only the filename
    wavfile_chirps = list(anechoic_chirps['filename'])
    wavfile_silence = list(anechoic_silence['filename'])
    return dataset_note, wavfile_chirps, wavfile_silence


def built_anechoeic_hdf5_dataset(wavfile_chirps, path_to_anechoic_dataset, wavfile_silence):
    # create the file
    f = h5py.File(path_to_anechoic_dataset, "w")
    # open it in append mode
    f = h5py.File(path_to_anechoic_dataset, 'a')

    for wavefile in tqdm(wavfile_chirps):
        print('Processing', wavefile)
        # extract the file from the zip and save it temporarely
        filename = 'room-000000/' + wavefile + '.wav'
        path_to_wavfile = get_zipped_file(filename, path_to_session_data, path_to_tmp)
        # get the signal from the tmp file
        wav, fs = sf.read(path_to_wavfile)
        f.create_dataset('recordings/48k/' + wavefile, data=wav)


    for wavefile in wavfile_silence:
        print('Processing', wavefile)
        # extract the file from the zip and save it temporarely
        filename = 'room-000000/' + wavefile + '.wav'
        path_to_wavfile = get_zipped_file(filename, path_to_session_data, path_to_tmp)
        # get the signal from the tmp file
        wav, fs = sf.read(path_to_wavfile)
        f.create_dataset('silence/48k/' + wavefile, data=wav)

    return path_to_anechoic_dataset


def build_rir_hdf5(wavfile_chirps, path_to_anechoic_dataset, path_to_anechoic_dataset_rir):
    f_raw = h5py.File(path_to_anechoic_dataset, 'r')
    f_rir = h5py.File(path_to_anechoic_dataset_rir, 'w')
    f_rir = h5py.File(path_to_anechoic_dataset_rir, 'a')

    # recreate the probe signal
    Fs = 48000
    ps = ProbeSignal('exp_sine_sweep', Fs)
    n_seconds = 10
    amplitude = 0.7
    n_repetitions = 3
    silence_at_start = 2
    silence_at_end = 2
    sweeprange = [100, 14e3]
    times, signal = ps.generate(n_seconds, amplitude, n_repetitions,
                                silence_at_start, silence_at_end, sweeprange)

    for wavefile in wavfile_chirps:
        x = f_raw['recordings/48k/' + wavefile]

        # compute the global delay from the playback:
        playback = x[:, -1]
        delay_sample = ps.compute_delay(playback[:, None], start=1, duration=10)
        delay = delay_sample/Fs

        try:
            assert delay > 0
        except:
            plt.plot(normalize(playback))
            plt.plot(normalize(ps.signal))
            plt.show()

        # estimate the rir in the remaining channels:
        for i in tqdm(range(30)):
            recording = x[:, i]
            rir_i = ps.compute_rir(recording[:, None])
            # for anechoic signal crop after 1 second
            rir_i = rir_i[0:int(0.5*Fs)]
            # store info in the anechoic dataset
            f_rir.create_dataset('rir/%s/%d' % (wavefile, i), data=rir_i)
            f_rir.create_dataset('delay/%s/%d' % (wavefile, i), data=delay)
    return


def compute_distances_from_rirs(path_to_anechoic_dataset_rir, dataset_note):
    f_rir = h5py.File(path_to_anechoic_dataset_rir, 'r')
    I = 30
    J = len(wavfile_chirps)

    Fs = 48000
    T = 24
    speed_of_sound = 331.3 + 0.606 * T

    src_mic_dist = np.zeros([I, J])
    L = int(0.5*Fs)
    rirs = np.zeros([L, I*J])
    direct_path_positions = np.zeros([I*J])

    ij = 0
    for j in tqdm(range(J)):
        for i in range(30):
            wavefile = wavfile_chirps[j]
            src_id = dataset_note.loc[dataset_note['filename'] == wavefile]['id']
            mic_id = j+33

            rir = f_rir['rir/%s/%d' % (wavefile, i)][()]
            rir = np.abs(rir).squeeze()
            rir = rir/np.max(rir)
            rirs[:, ij] = rir

            peaks, _ = sg.find_peaks(rir, height=0.2, distance=50, width=2, prominence=0.6)
            # plt.plot(rir)
            # plt.plot(peaks, rir[peaks], "x")
            # plt.plot(np.zeros_like(rir), "--", color="gray")
            # plt.show()

            direct_path_positions[ij] = np.min(peaks)
            recording_offset = f_rir['delay/%s/%d' % (wavefile, i)][()]
            toa = direct_path_positions[ij]/Fs
            print(recording_offset)

            fig, ax = plt.subplots()
            newax1 = ax.twiny()
            newax2 = ax.twiny()
            fig.subplots_adjust(bottom=0.40)

            newax1.set_frame_on(True)
            newax2.set_frame_on(True)
            newax1.patch.set_visible(False)
            newax2.patch.set_visible(False)
            newax1.xaxis.set_ticks_position('bottom')
            newax2.xaxis.set_ticks_position('bottom')
            newax1.xaxis.set_label_position('bottom')
            newax2.xaxis.set_label_position('bottom')
            newax1.spines['bottom'].set_position(('outward', 40))
            newax2.spines['bottom'].set_position(('outward', 80))

            ax.plot(np.arange(len(rir)), rir)
            ax.axvline(recording_offset*Fs)
            newax1.plot(np.arange(len(rir))/Fs, rir)
            newax2.plot(np.arange(len(rir))/Fs*speed_of_sound, rir)

            ax.set_xlabel('Time [samples]')
            newax1.set_xlabel('Time [seconds]')
            newax2.set_xlabel('Distance [meter]')

            plt.show()

            src_mic_dist[i, j] = (toa - recording_offset)*speed_of_sound
            ij += 1

    plt.imshow(rirs, extent=[0, I*J, 0, L], aspect='auto')
    for j in range(J):
        plt.axvline(j*30)
    plt.scatter(np.arange(I*J)+0.5, L-direct_path_positions, c='C1')
    plt.tight_layout()
    plt.show()

    return src_mic_dist


if __name__ == "__main__":

    dataset_dir = './data/dECHORATE/'
    path_to_tmp = '/tmp/'
    path_to_processed = './data/processed/'

    session_filename = "recordings/room-000000.zip"
    path_to_session_data = dataset_dir + session_filename

    ## GET FILENAMES FROM THE CSV ANNOTATION DATABASE
    note_filename = 'annotations/dECHORATE_recordings_note.csv'
    path_to_note = dataset_dir + note_filename
    dataset_note, wavfile_chirps, wavfile_silence = silence_and_chirps_names(path_to_note)

    ## MAKE JUST THE HDF5 ANECHOIC DATASET
    path_to_anechoic_dataset = path_to_processed + 'anechoic_raw_data.hdf5'
    # built_anechoeic_hdf5_dataset(wavfile_chirps, path_to_anechoic_dataset, wavfile_silence)

    ## DECONVOLVE THE CHIRPS
    path_to_anechoic_dataset_rir = path_to_processed + 'anechoic_rir_data.hdf5'
    # build_rir_hdf5(wavfile_chirps, path_to_anechoic_dataset, path_to_anechoic_dataset_rir)

    ## COMPUTE MICROPHONES-SOURCE DISTANCES
    src_mic_dist = compute_distances_from_rirs(
        path_to_anechoic_dataset_rir, dataset_note)

    plt.imshow(src_mic_dist)
    plt.show()
    pass
