import os
import h5py
import numpy as np
import scipy as sp
import scipy.signal as sg
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt

from shutil import rmtree
from zipfile import ZipFile
from tqdm import tqdm

import matplotlib.pyplot as plt

from src.stimulus import ProbeSignal
from src.utils.dsp_utils import *

# opening the zip file in READ mode


def get_zipfile_list(path_to_file):
    with ZipFile(path_to_file, 'r') as zip:
        return zip.namelist()


def get_zipped_file(filename, path_to_zipfile, path_to_output):
    with ZipFile(path_to_zipfile, 'r') as zip:
        return zip.extract(filename, path_to_output)


def get_signal_note_and_filenames(path_to_note, session_id, signal, src_id):
    f, c, w, s, e, n = [int(i) for i in list(session_id)]
    dataset_note = pd.read_csv(path_to_note)
    # select the annechoic recordings and only chirps
    signal_note = dataset_note.loc[
        (dataset_note['floor'] == f)
        & (dataset_note['ceiling'] == c)
        & (dataset_note['west'] == w)
        & (dataset_note['east'] == e)
        & (dataset_note['north'] == n)
        & (dataset_note['south'] == s)
        & (dataset_note['fornitures'] == False)
        & (dataset_note['id'] == src_id)
        & (dataset_note['signal'] == signal)
    ]
    try:
        assert len(signal_note) == 1
    except:
        print(signal_note)

    # check that there are not detrimental artifacts
    try:
        assert np.all(signal_note['artifacts'] < 2)
    except:
        print('Artifacts in')
        print(chirps['artifacts'])
    try:
        assert np.all(signal_note['artifacts'] < 2)
    except:
        print(silence['artifacts'])

    # get only the filename
    wavfile = list(signal_note['filename'])
    return signal_note, wavfile[0]


def extract_raw_data(wavefile, session_id, path_to_zipfile, path_to_extracted, mic_id, signal):

    print('Processing', wavefile)
    # extract the file from the zip and save it temporarely
    filename = 'room-'+session_id+'/' + wavefile + '.wav'
    path_to_wavfile = get_zipped_file(
        filename, path_to_zipfile, path_to_extracted)

    path_to_extracted = path_to_wavfile.replace('.wav', '_%s_mic-%d.wav' % (signal, mic_id))
    wav, fs = sf.read(path_to_wavfile)
    sf.write(path_to_extracted, wav[:, mic_id-1], fs)

    return path_to_extracted


def get_rir(path_to_extracted, windowing):

    if 'chirp' in path_to_extracted:
        # recreate the probe signal
        Fs = 48000
        ps = ProbeSignal('exp_sine_sweep', Fs)
        n_seconds = 10
        amplitude = 0.7
        n_repetitions = 3
        silence_at_start = 2
        silence_at_end = 2
        sweeprange = [100, 14e3]
        ps.generate(n_seconds, amplitude, n_repetitions,
                    silence_at_start, silence_at_end, sweeprange)

        recording, Fs_rec = sf.read(path_to_extracted)
        assert Fs == Fs_rec
        assert len(recording.shape) == 1

        rir = ps.compute_rir(recording[:, None], windowing=windowing)
        # for anechoic signal crop after 1 second
        rir = rir[0:Fs]

        suffix = '_wrir.wav' if windowing else '_rir.wav'
        filename = path_to_extracted = path_to_extracted.replace('.wav', suffix)
        sf.write(filename, rir, Fs)

        return rir


if __name__ == "__main__":

    dataset_dir = './data/dECHORATE/'
    path_to_tmp = '/tmp/'
    path_to_processed = './data/processed/'

    Fs = 48000

    for signal in ['chirp']:
        session_id = '011110'
        mic_id = 1
        src_id = 2

        session_filename = "recordings/room-%s.zip" % session_id
        path_to_session_data = dataset_dir + session_filename

        ## GET FILENAMES FROM THE CSV ANNOTATION DATABASE
        print('Getting filenames...')
        note_filename = 'annotations/dECHORATE_recordings_note.csv'
        path_to_note = dataset_dir + note_filename
        signal_note, wavfile = get_signal_note_and_filenames(
            path_to_note, session_id, signal, src_id)
        print(signal_note)
        print(wavfile)

        ## MAKE JUST THE HDF5 ANECHOIC DATASET
        print('Extracting the raw data...')
        path_to_file = './data/tmp/'
        path_to_extracted = extract_raw_data(wavfile, session_id,
                        path_to_session_data, path_to_file, mic_id, signal)

        # ## DECONVOLVE THE CHIRPS
        # print('Estimating RIRs...')
        if signal == 'chirp':
            for windowing in [True, False]:
                rir = get_rir(path_to_extracted, windowing)

                plt.subplot(211)
                plt.plot(np.abs(rir), label='wrir' if windowing else 'rir')

                # stft
                plt.subplot(212)
                freqs, times, RIR = sg.stft(rir, fs=Fs)
                RIR = np.mean(RIR, axis=1)
                plt.plot(np.abs(RIR))


            # if signal == 'noise':
            #     noise, Fs = sf.read(path_to_extracted)

            #     freqs, times, NOISE = sg.stft(noise, fs=Fs, nperseg=2048)

            #     axarr[1].imshow(np.log(np.abs(NOISE)))

        plt.legend()
        plt.show()


    pass
