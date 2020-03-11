import os
import h5py
import numpy as np
import scipy.signal as sg
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt

from shutil import rmtree
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


def silence_and_chirps_names(path_to_note, session_id):
    f, c, w, s, e, n = [int(i) for i in list(session_id)]
    dataset_note = pd.read_csv(path_to_note)
    # select the annechoic recordings and only chirps
    chirps = dataset_note.loc[
        (dataset_note['floor'] == f)
        & (dataset_note['ceiling'] == c)
        & (dataset_note['west'] == w)
        & (dataset_note['east'] == e)
        & (dataset_note['north'] == n)
        & (dataset_note['south'] == s)
        & (dataset_note['fornitures'] == False)
        & (dataset_note['signal'] == 'chirp')
    ]

    silence = dataset_note.loc[
        (dataset_note['floor'] == f)
        & (dataset_note['ceiling'] == c)
        & (dataset_note['west'] == w)
        & (dataset_note['east'] == e)
        & (dataset_note['north'] == n)
        & (dataset_note['south'] == s)
        & (dataset_note['fornitures'] == False)
        & (dataset_note['signal'] == 'silence')
    ]
    # check that there are not detrimental artifacts
    try:
        assert np.all(chirps['artifacts'] < 2)
    except:
        print('Artifacts in')
        print(chirps['artifacts'])
    try:
        assert np.all(silence['artifacts'] < 2)
    except:
        print(silence['artifacts'])

    # get only the filename
    wavfile_chirps = list(chirps['filename'])
    wavfile_silence = list(silence['filename'])
    return dataset_note, wavfile_chirps, wavfile_silence


def built_anechoeic_hdf5_dataset(wavfile_chirps, path_to_anechoic_dataset, wavfile_silence, session_id):
    # create the file
    f = h5py.File(path_to_anechoic_dataset, "w")
    # open it in append mode
    f = h5py.File(path_to_anechoic_dataset, 'a')

    for wavefile in tqdm(wavfile_chirps):
        print('Processing', wavefile)
        # extract the file from the zip and save it temporarely
        filename = 'room-'+session_id+'/' + wavefile + '.wav'
        path_to_wavfile = get_zipped_file(filename, path_to_session_data, path_to_tmp)
        # get the signal from the tmp file
        wav, fs = sf.read(path_to_wavfile)
        f.create_dataset('recordings/48k/' + wavefile, data=wav)


    for wavefile in wavfile_silence:
        print('Processing', wavefile)
        # extract the file from the zip and save it temporarely
        filename = 'room-'+session_id+'/' + wavefile + '.wav'
        path_to_wavfile = get_zipped_file(filename, path_to_session_data, path_to_tmp)
        # get the signal from the tmp file
        wav, fs = sf.read(path_to_wavfile)
        f.create_dataset('silence/48k/' + wavefile, data=wav)

    rmtree(path_to_tmp + 'room-' + session_id)

    return path_to_anechoic_dataset


def build_rir_hdf5(wavfile_chirps, path_to_anechoic_dataset, path_to_anechoic_dataset_rir, session_id):
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

        assert x.shape[1] == 31
        I = 30

        # compute the global delay from the playback:
        playback = x[:, -1]
        rir_playback = ps.compute_rir(playback[:, None], windowing=False)
        f_rir.create_dataset('rir/%s/playback/' % (wavefile), data=rir_playback)

        delay_sample = np.argmax(np.abs(rir_playback))
        delay_second = delay_sample/Fs
        print('Delay sample', delay_sample)

        plt.subplot(211)
        plt.plot(times, signal, label='Probe')
        plt.plot(np.arange(len(playback))/Fs, playback, alpha=0.5, label='Playback')
        plt.legend(loc=1)
        plt.subplot(212)
        plt.plot(np.arange(len(rir_playback))/Fs, rir_playback, alpha=0.8, label='playback')
        plt.legend(loc=1)
        plt.suptitle('Playback. Delay %d => %1.3f ms' % (delay_sample, delay_second*1000))
        plt.savefig('./reports/figures/rirs/room-%s/%s_%s_playback.png' % (session_id, session_id, wavefile))
        plt.close()


        try:
            assert delay > 0
            assert int(delay_sample) == 4444
        except:
            print('Delay assert Error', wavefile, 'with delay', delay_sample)

        # estimate the rir in the remaining channels:
        for i in tqdm(range(I)):
            recording = x[:, i]

            rir_i = ps.compute_rir(recording[:, None], windowing=False)
            # for anechoic signal crop after 0.5 second
            rir_i = rir_i[0:int(0.5*Fs)]

            plt.subplot(211)
            plt.plot(np.arange(len(signal))/Fs, signal, label='Probe')
            plt.plot(np.arange(len(recording))/Fs, recording, alpha=0.5, label='Recording')
            plt.legend(loc=1)
            plt.subplot(212)
            plt.plot(np.arange(len(rir_i))/Fs, rir_i, alpha=0.8, label='RIR')
            plt.legend(loc=1)
            plt.suptitle('RIR mic %d. Delay %d => %1.3f ms' % (i, delay_sample, delay_second*1000))
            plt.savefig('./reports/figures/rirs/room-%s/%s_%s_mic-%d.png' % (session_id, session_id, wavefile, i))
            plt.close()

            # store info in the anechoic dataset
            f_rir.create_dataset('rir/%s/%d' % (wavefile, i), data=rir_i)
            f_rir.create_dataset('delay_sample/%s/%d' % (wavefile, i), data=delay_sample)
            f_rir.create_dataset('delay_second/%s/%d' % (wavefile, i), data=delay_second)

    return


if __name__ == "__main__":

    dataset_dir = './data/dECHORATE/'
    path_to_tmp = '/tmp/'
    path_to_processed = './data/processed/'

    session_id = '000001'
    session_filename = "recordings/room-%s.zip" % session_id
    path_to_session_data = dataset_dir + session_filename

    ## GET FILENAMES FROM THE CSV ANNOTATION DATABASE
    print('Getting filenames...')
    note_filename = 'annotations/dECHORATE_recordings_note.csv'
    path_to_note = dataset_dir + note_filename
    dataset_note, wavfile_chirps, wavfile_silence = silence_and_chirps_names(path_to_note, session_id)

    ## MAKE JUST THE HDF5 ANECHOIC DATASET
    print('Extracting the raw data...')
    path_to_anechoic_dataset = path_to_processed + '%s_raw_data.hdf5' % session_id
    built_anechoeic_hdf5_dataset(wavfile_chirps, path_to_anechoic_dataset, wavfile_silence, session_id)

    ## DECONVOLVE THE CHIRPS
    print('Estimating RIRs...')
    path_to_anechoic_dataset_rir = path_to_processed + '%s_rir_data.hdf5' % session_id
    build_rir_hdf5(wavfile_chirps, path_to_anechoic_dataset, path_to_anechoic_dataset_rir, session_id)


    pass
