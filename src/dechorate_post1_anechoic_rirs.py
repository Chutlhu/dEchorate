import os
import h5py
import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt

from zipfile import ZipFile

from src.stimulus import ProbeSignal

dataset_dir = './data/dECHORATE/'
path_to_tmp = '/tmp/'
path_to_processed = './data/processed/'

# specifying the zip file name
session_filename = "recordings/room-000000.zip"
path_to_session_data = dataset_dir + session_filename

# opening the zip file in READ mode
def get_zipfile_list(path_to_file):
    with ZipFile(path_to_file, 'r') as zip:
        return zip.namelist()


def get_zipped_file(filename, path_to_zipfile, path_to_output):
    with ZipFile(path_to_zipfile, 'r') as zip:
        return zip.extract(filename, path_to_output)

## GET FILENAMES FROM THE CSV ANNOTATION DATABASE
note_filename = 'annotations/dECHORATE_recordings_note.csv'
path_to_note = dataset_dir + note_filename
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
print(wavfile_chirps)
print(wavfile_silence)

## MAKE JUST THE HDF5 ANECHOIC DATASET
path_to_anechoic_dataset = path_to_processed + 'anechoic_data.hdf5'
# create the file
# f = h5py.File(path_to_anechoic_dataset, "w")
# # open it in append mode
# f = h5py.File(path_to_anechoic_dataset, 'a')

# for wavefile in wavfile_chirps:
#     print('Processing', wavefile)
#     # extract the file from the zip and save it temporarely
#     filename = 'room-000000/' + wavefile + '.wav'
#     path_to_wavfile = get_zipped_file(filename, path_to_session_data, path_to_tmp)
#     # get the signal from the tmp file
#     wav, fs = sf.read(path_to_wavfile)
#     f.create_dataset('recordings/48k/' + wavefile, data=wav)


# for wavefile in wavfile_silence:
#     print('Processing', wavefile)
#     # extract the file from the zip and save it temporarely
#     filename = 'room-000000/' + wavefile + '.wav'
#     path_to_wavfile = get_zipped_file(filename, path_to_session_data, path_to_tmp)
#     # get the signal from the tmp file
#     wav, fs = sf.read(path_to_wavfile)
#     f.create_dataset('silence/48k/' + wavefile, data=wav)

## DECONVOLVE THE CHIRPS
f = h5py.File(path_to_anechoic_dataset, 'r')
wavefile = wavfile_chirps[0]
x = f['recordings/48k/' + wavefile]

playback = x[:, -1]
recording = x[:, 0]

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

ps.compute_delay(recording, start=2, duration=1)

# plt.plot(signal)
# plt.plot(playback)
# plt.show()
# h = ps.compute_rir(rec)
# print(x.shape)
1/0
