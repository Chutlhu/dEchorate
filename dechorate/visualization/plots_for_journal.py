import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pyroomacoustics as pra
import peakutils as pk

from copy import deepcopy as cp
from datetime import datetime

from matplotlib.widgets import Slider, RadioButtons, Button, TextBox, CheckButtons
from mpl_toolkits.mplot3d import Axes3D

from dechorate import constants
from dechorate.dataset import SyntheticDataset, DechorateDataset
from dechorate.utils.file_utils import load_from_pickle, save_to_pickle
from dechorate.utils.dsp_utils import normalize

from tkinter.filedialog import askopenfilename

dataset_dir = './data/dECHORATE/'
path_to_processed = './data/processed/'
path_to_note_csv = dataset_dir + 'annotations/dECHORATE_database.csv'

params = {
    'Fs' : constants['Fs'],
}
datasets = constants['datasets']
curr_reflectors = constants['refl_order_calibr'][:7]
refl_order = constants['refl_order_pyroom']
Fs = constants['Fs']
recording_offset = constants['recording_offset']
speed_of_sound = constants['speed_of_sound']

# data to visualize
dataset_id = '011111'

## INITIALIZE FIGURE


rirs = np.load('./data/interim/all_rirs_9srcs.npy')

## IMPORT DATA
dataset_dir = './data/dECHORATE/'
path_to_processed = './data/processed/'
path_to_note_csv = dataset_dir + 'annotations/dECHORATE_database.csv'
path_to_after_calibration = path_to_processed + 'post2_calibration/calib_output_mics_srcs_pos.pkl'
note_dict = load_from_pickle(path_to_after_calibration)

toa = note_dict['toa_pck']

L, I, J, D = rirs.shape
K, I, J = toa.shape


i, j = 23, 0

print(datasets)
didx = np.array([0,1,2,3,5])

fig, axarr = plt.subplots(2,1)

for d in didx:

    rir = np.abs(normalize(rirs[:, i, j, d]))

    axarr[0].plot(np.arange(L)/Fs, rir, label=datasets[d])

    a = 288
    b = 384
    if d == 0:
        axarr[0].fill_between(np.arange(a, b)/Fs, -0.2, 1.2, color='C8', alpha=0.2)

    dp = rir[a:b]
    p = np.argmax(np.abs(dp))
    dp_deconv = np.real(np.fft.ifft(np.fft.fft(rir, L) / np.fft.fft(dp, L)))[:L-p]
    # restore the direct path
    dp_deconv = np.concatenate([np.zeros(p), dp_deconv])
    echogram = np.abs(normalize(dp_deconv))
    axarr[1].plot(np.arange(L)/Fs, echogram, label=datasets[d])

for k in range(K):
    if k == 0:
        order = 0
    if k > 0 and k < 8:
        order = 1
    axarr[0].axvline(x=toa[k, i, j], ls='--')
    axarr[1].axvline(x=toa[k, i, j], ls='--')
    axarr[0].annotate(r'$\leftarrow \tau_{%d}^{%d}$' % (k, order), [toa[k, i, j], 0.9])
    axarr[1].annotate(r'$\leftarrow \tau_{%d}^{%d}$' % (k, order), [toa[k, i, j], 0.9])

axarr[0].set_xlim([toa[0, i, j]-0.001, toa[0, i, j] + 0.02])
axarr[0].legend()
axarr[1].set_xlim([toa[0, i, j]-0.001, toa[0, i, j] + 0.02])
axarr[1].legend()
plt.show()


1/0

fig, axarr = plt.subplots(2, 1, figsize=(12,9))
for idd, d in enumerate([0, 1]):
    for idf in [0, 1]:

        rir = all_rirs[:, i, j, d]
        echogram = np.abs(normalize(rir))

        if idf == 0:
            pass
        else:
            # take the dp from... anechoic
            a = 200
            b = 333
            dp = rir[a:b]
            p = np.argmax(np.abs(dp))
            dp_deconv = np.real(np.fft.ifft(np.fft.fft(rir, L) / np.fft.fft(dp, L)))[:L-p]
            # restore the direct path
            dp_deconv = np.concatenate([np.zeros(p), dp_deconv])
            echogram = np.abs(normalize(dp_deconv))

        L = len(rir)

        # axarr[idf].plot(np.arange(L)/Fs, echogram)
        axarr[idf].plot(echogram)
        axarr[idf].set_title('Dataset %s, Source %d' % (datasets[d], j))
        # axarr[idf].set_xlim([0.005, 0.02])


plt.tight_layout()
plt.show()
