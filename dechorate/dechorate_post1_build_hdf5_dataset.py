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


def get_wavefile_from_database(path_to_note, signals, mic_ids, src_ids, room_codes):
    dataset_note = pd.read_csv(path_to_note)
    for room_code in room_codes:
        for signal in signals:
            for mic_id in mic_ids:
                for src_id in src_ids:

                    print(room_code, signal, mic_id, src_id)

                    # select the annechoic recordings and only chirps
                    wavefiles_db = dataset_note.loc[
                          (dataset_note['room_rfl_floor']   == int(room_code[0]))
                        & (dataset_note['room_rfl_ceiling'] == int(room_code[1]))
                        & (dataset_note['room_rfl_west']    == int(room_code[2]))
                        & (dataset_note['room_rfl_east']    == int(room_code[3]))
                        & (dataset_note['room_rfl_north']   == int(room_code[4]))
                        & (dataset_note['room_rfl_south']   == int(room_code[5]))
                        & (dataset_note['room_rfl_south']   == int(room_code[5]))
                        & (dataset_note['src_id'] == src_id)
                        & (dataset_note['mic_id'] == mic_id)
                        & (dataset_note['src_signal'] == signal)
                    ]
                    # check that there are not detrimental artifacts
                    assert np.all(wavefiles_db['rec_artifacts'] < 2)
                    # check that there is only one filename
                    assert len(wavefiles_db) == 1
                    # get only the filename
                    filename = str(wavefiles_db['filename'].values[0])
                    yield (filename, signal, mic_id, src_id, room_code)


def wave_loader(wavefile, session_filename, path_to_session_zip_dir, path_to_output):

    # for wavefile in tqdm(wavefiles):
    print('Processing', wavefile, 'in', session_filename)
    # extract the file from the zip and save it
    filename = os.path.join(session_filename, wavefile + '.wav')
    path_to_current_wavefile = get_zipped_file(filename, path_to_session_zip_dir, path_to_output)
    # path_to_wavefiles.append(path_to_current_wavefile)
    wav, fs = sf.read(path_to_current_wavefile)
    assert fs == constants['Fs']
    return wav


def build_rir_hdf5(wavefile_chirps, path_to_output_anechoic_dataset, path_to_output_anechoic_dataset_rir):
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

    for wavefile in wavefile_chirps:
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
    src_ids = constants['src_ids']
    mic_ids = constants['mic_ids']
    signals = ['chirp']

    # # INITIALIZE THE HDF5 DATASET

    # DATASET structure: room x mics x srcs x signal + bonus
    # room = [000000, 010000, ..., 111111, 0F000F]
    # mics = [0, ..., 29, 30] : 0-29 capsule, 30th loopback
    # srcs = [0, ..., 9] : 0-3 dir, 4-6 omni
    # signal = ['chirp', 'silence', 'speech', 'noise', ..., 'RIR']
    # bonus: book for polarity

    path_to_output_dataset_hdf5 = os.path.join(path_to_output, 'dechorate.hdf5')
    f = h5py.File(path_to_output_dataset_hdf5, "w")

    # open it in append mode
    with h5py.File(path_to_output_dataset_hdf5, 'a') as data_file:
        ## GET FILENAMES FROM THE CSV ANNOTATION DATABASE
        path_to_note = os.path.join(cwd, 'data', 'final', 'manual_annatotion.csv')
        mics_here = [1] # the wavefile collects all the channels already
        for (wavename, signal, mic_id, src_id, room_code) in get_wavefile_from_database(path_to_note, signals, [1], src_ids, room_codes):
            session_filename = 'room-%s' % room_code
            path_to_session_zip = os.path.join(path_to_recordings, session_filename+'.zip')
            wav = wave_loader(wavename, session_filename, path_to_session_zip, path_to_output)

            group = '/%s/%s/%d/%d' % ()
            data_file.create_dataset(room_code, data=wav)



    # # wavefiles_silence = get_signal_filename_from_database(path_to_note, 'silence', 1, room_code)


    # for wav in wave_loader(wavefiles_chirps, session_filename, path_to_session_zip, path_to_output):
    #     print(wav.shape)
    #     1/0
    # path_to_wavefiles_chirp = built_rec_hdf5(wavefiles_chirps, session_filename, path_to_session_zip, path_to_output)
    # path_to_wavefiles_silence = built_rec_hdf5(wavefiles_silence, session_filename, path_to_session_zip, path_to_output)


    # ## DECONVOLVE THE CHIRPS
    # path_to_output_anechoic_dataset_rir = path_to_output + 'anechoic_rir_data.hdf5'
    # build_rir_hdf5(wavefile_chirps, path_to_output_anechoic_dataset, path_to_output_anechoic_dataset_rir)


    pass
