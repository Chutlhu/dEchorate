import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pyroomacoustics as pra

from copy import deepcopy as cp

from matplotlib.widgets import Slider, RadioButtons, Button, TextBox
from mpl_toolkits.mplot3d import Axes3D

from src.dataset import SyntheticDataset, DechorateDataset
from src.utils.file_utils import load_from_pickle, save_to_pickle, save_to_matlab
from src.utils.dsp_utils import normalize


dataset_dir = './data/dECHORATE/'
path_to_processed = './data/processed/'
path_to_note_csv = dataset_dir + 'annotations/dECHORATE_database.csv'

imag_dset = SyntheticDataset()
real_dset = DechorateDataset(path_to_processed, path_to_note_csv)

params = {
    'Fs' : 48000,
}

i, j = (1, 0)

imag_dset.set_room_size([5.741, 5.763, 2.353])
imag_dset.set_c(343)
imag_dset.set_k_order(4)
imag_dset.set_k_reflc(15)
datasets = ['000000', '010000', '011000', '011100', '011110', '011111',
            '001000', '000100', '000010', '000001']

real_dset.set_dataset('000000')
real_dset.set_entry(i, j)
mic_pos, src_pos = real_dset.get_mic_and_src_pos()
times, h_rec = real_dset.get_rir()

print('mic_pos init', mic_pos)
print('src_pos init', src_pos)

all_rirs = np.load('./data/tmp/all_rirs.npy')
all_rirs_clean = np.load('./data/tmp/all_rirs_clean.npy')
toa_note = load_from_pickle('./data/tmp/toa_note.pkl')

save_to_matlab('./data/interim/manual_annotation/all_rirs.mat', {'all_rirs': all_rirs})
save_to_matlab('./data/interim/manual_annotation/all_rirs_clean.mat', {'all_rirs_clean': all_rirs_clean})
save_to_matlab('./data/interim/manual_annotation/toa_note.mat', toa_note)

# L, I, J, D = all_rirs.shape
# K, I, J, D = toa_note['toa'].shape
# Fs = params['Fs']

# taus_list = [r'%d $\tau_{%s}^{%d}$' % (k, toa_note['wall'][k, i, j, 0].decode(), toa_note['order'][k, i, j, 0]) for k in range(K)]

# print(toa_note.keys())

# i, j = 0, 0
# L = 5000
# rirs = np.clip(all_rirs_clean[:L, i, j, :], 0, 0.4)
# tofs_rir = toa_note['toa'][:, i, j, :]

# for d in range(D):
#     rirs[:, d] = rirs[:, 0] * (1 - np.exp(-(np.arange(L))/1500))

# plt.imshow(rirs, extent=[0, D, 0, L], aspect='auto')
# plt.scatter(np.arange(D)+0.5, tofs_rir.T.flatten()*Fs,
#             c='C1', label='Peak Picking')
# # plt.scatter(np.arange(I*J)+0.5, L - recording_offset -
# #             tofs_simulation.T.flatten()*Fs, c='C2', label='Pyroom')
# plt.tight_layout()
# plt.legend()
# plt.show()
