import os
import h5py
import numpy as np
import scipy.signal as sg
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt

from zipfile import ZipFile
from tqdm import tqdm

from dechorate.stimulus import ProbeSignal
from dechorate.utils.dsp_utils import *

# opening the zip file in READ mode
def get_zipfile_list(path_to_file):
    with ZipFile(path_to_file, 'r') as zip:
        return zip.namelist()


def get_zipped_file(filename, path_to_zipfile, path_to_output):
    with ZipFile(path_to_zipfile, 'r') as zip:
        return zip.extract(filename, path_to_output)


def get_signal_filename_from_database(path_to_note, signal, code='000000'):
    dataset_note = pd.read_csv(path_to_note)
    # select the annechoic recordings and only chirps
    wavefiles_db = dataset_note.loc[
          (dataset_note['room_rfl_floor']   == int(code[0]))
        & (dataset_note['room_rfl_ceiling'] == int(code[1]))
        & (dataset_note['room_rfl_west']    == int(code[2]))
        & (dataset_note['room_rfl_east']    == int(code[3]))
        & (dataset_note['room_rfl_north']   == int(code[4]))
        & (dataset_note['room_rfl_south']   == int(code[5]))
        & (dataset_note['src_signal'] == signal)
    ]
    # check that there are not detrimental artifacts
    assert np.all(wavefiles_db['rec_artifacts'] < 2)
    # get only the filename
    wavfiles = list(wavefiles_db['filename'])
    return wavfiles


def built_rec_hdf5(wavfile_chirps, session_filename, path_to_session_zip_dir, path_to_output_anechoic_dataset):

    # create the file
    f = h5py.File(path_to_output_anechoic_dataset, "w")
    # open it in append mode
    f = h5py.File(path_to_output_anechoic_dataset, 'a')

    for wavefile in tqdm(wavfile_chirps):
        print('Processing', wavefile)
        # extract the file from the zip and save it temporarely
        filename = session_filename + '.zip/' + wavefile + '.wav'
        path_to_wavfile = get_zipped_file(filename, path_to_session_zip, '/tmp/')
        # get the signal from the tmp file
        wav, fs = sf.read(path_to_wavfile)
        f.create_dataset('recordings/48k/' + wavefile, data=wav)
        1/0


    # for wavefile in wavfile_silence:
    #     print('Processing', wavefile)
    #     # extract the file from the zip and save it temporarely
    #     filename = session_filename + '.zip/' + wavefile + '.wav'
    #     path_to_wavfile = get_zipped_file(filename, path_to_session_data, '/tmp/')
    #     # get the signal from the tmp file
    #     wav, fs = sf.read(path_to_wavfile)
    #     f.create_dataset('silence/48k/' + wavefile, data=wav)

    return path_to_output_anechoic_dataset


def build_rir_hdf5(wavfile_chirps, path_to_output_anechoic_dataset, path_to_output_anechoic_dataset_rir):
    f_raw = h5py.File(path_to_output_anechoic_dataset, 'r')
    f_rir = h5py.File(path_to_output_anechoic_dataset_rir, 'w')
    f_rir = h5py.File(path_to_output_anechoic_dataset_rir, 'a')

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
        rir_playback = ps.compute_rir(playback[:, None])
        delay_sample = np.argmax(np.abs(rir_playback))
        # delay_sample = ps.compute_delay(playback[:, None], start=1, duration=10)
        delay = delay_sample/Fs
        print(delay_sample)

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
            f_rir.create_dataset('delay/%s/%d' % (wavefile, i), data=delay_sample)
            plt.show()
    return


if __name__ == "__main__":

    path_to_tmp = '/tmp/'
    path_to_output = './data/final/'

    path_to_recordings = './data/dECHORATE/recordings/'

    # only anechoic data
    code = "000000"
    session_filename = "room-%s"%code # anechoic data

    ## GET FILENAMES FROM THE CSV ANNOTATION DATABASE
    path_to_note = './data/final/manual_annatotion.csv'
    wavfiles_chirps = get_signal_filename_from_database(path_to_note, 'chirp', code=code)
    wavfiles_silence = get_signal_filename_from_database(path_to_note, 'silence', code=code)

    ## MAKE JUST THE HDF5 ANECHOIC DATASET
    path_to_session_zip = path_to_recordings + session_filename + '.zip'
    path_to_output_anechoic_dataset = path_to_output + session_filename + '_recorded_chirp.hdf5'
    built_rec_hdf5(wavfiles_chirps, path_to_session_zip, path_to_output_anechoic_dataset)

    # ## DECONVOLVE THE CHIRPS
    # path_to_output_anechoic_dataset_rir = path_to_output + 'anechoic_rir_data.hdf5'
    # build_rir_hdf5(wavfile_chirps, path_to_output_anechoic_dataset, path_to_output_anechoic_dataset_rir)


    pass
