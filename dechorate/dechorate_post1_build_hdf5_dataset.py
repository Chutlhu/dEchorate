import os
import h5py
import tempfile
import numpy as np
import scipy.signal as sg
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt

from zipfile import ZipFile
from tqdm import tqdm

from dechorate import constants # these are stored in the __init__py file
from dechorate.stimulus import ProbeSignal
from dechorate.utils.dsp_utils import *



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
        & (dataset_note['room_rfl_south']   == int(code[5]))
        & (dataset_note['src_signal'] == signal)
    ]
    # check that there are not detrimental artifacts
    assert np.all(wavefiles_db['rec_artifacts'] < 2)
    # get only the filename
    wavfiles = list(wavefiles_db['filename'])
    # discard repetition
    wavfiles = list(set(wavfiles))
    return wavfiles


def built_rec_hdf5(wavfiles, session_filename, path_to_session_zip_dir, path_to_output):

    # # create the file
    # f = h5py.File(path_to_output_anechoic_dataset, "w")
    # # open it in append mode
    # f = h5py.File(path_to_output_anechoic_dataset, 'a')
    # tmpdir = tempfile.gettempdir()
    # # get the signal from the tmp file
    # wav, fs = sf.read(path_to_current_wavfile)
    # path_to_output_wav = os.path.join(path_to_output, wavefile)
    # f.create_dataset(path_to_output_wav, data=wav)

    path_to_wavefiles = []
    for wavefile in tqdm(wavfiles):
        print('Processing', wavefile, 'in', session_filename)
        # extract the file from the zip and save it
        filename = os.path.join(session_filename, wavefile + '.wav')
        path_to_current_wavfile = get_zipped_file(filename, path_to_session_zip, path_to_output)
        path_to_wavefiles.append(path_to_current_wavfile)

    return path_to_wavefiles


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

    cwd = os.getcwd()

    path_to_tmp = os.path.join('/', 'tmp')
    path_to_output = os.path.join(cwd, 'data', 'final')
    path_to_recordings = os.path.join(cwd, 'data', 'dECHORATE', 'recordings')

    room_codes = constants['datasets']
    Fs = constants['Fs']

    # # INITIALIZE THE HDF5 DATASET
    path_to_output_dataset_hdf5 = os.path.join(path_to_output, 'dechorate.hdf5')
    f = h5py.File(path_to_output_dataset_hdf5, "w")
    # open it in append mode
    f = h5py.File(path_to_output_dataset_hdf5, 'a')

    # DATASET structure: room x mics x srcs x signal + bonus
    # room = [000000, 010000, ..., 111111, 0F000F]
    # mics = [0, ..., 29, 30] : 0-29 capsule, 30th loopback
    # srcs = [0, ..., 9] : 0-3 dir, 4-6 omni
    # signal = ['chirp', 'silence', 'speech', 'noise', ..., 'RIR']
    # bonus: book for polarity

    for room_code in room_codes:

        session_filename = "room-%s"%room_code # anechoic data

        ## GET FILENAMES FROM THE CSV ANNOTATION DATABASE
        path_to_note = os.path.join(cwd, 'data', 'final', 'manual_annatotion.csv')
        wavfiles_chirps = get_signal_filename_from_database(path_to_note, 'chirp', code=room_code)
        wavfiles_silence = get_signal_filename_from_database(path_to_note, 'silence', code=room_code)

        ## MAKE JUST THE HDF5 ANECHOIC DATASET
        path_to_session_zip = os.path.join(path_to_recordings, session_filename + '.zip')
        # os.makedirs(path_to_output_anechoic_dataset,  exist_ok=True)
        path_to_wavefiles_chirp = built_rec_hdf5(wavfiles_chirps, session_filename, path_to_session_zip, path_to_output)
        path_to_wavefiles_silence = built_rec_hdf5(wavfiles_silence, session_filename, path_to_session_zip, path_to_output)

        # ## DECONVOLVE THE CHIRPS
        # path_to_output_anechoic_dataset_rir = path_to_output + 'anechoic_rir_data.hdf5'
        # build_rir_hdf5(wavfile_chirps, path_to_output_anechoic_dataset, path_to_output_anechoic_dataset_rir)


    pass
