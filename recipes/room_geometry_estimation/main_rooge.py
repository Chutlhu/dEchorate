import numpy as np
import peakutils as pk
import matplotlib.pyplot as plt

from scipy.spatial import distance

from src import constants
from src.dataset import DechorateDataset
from src.utils.file_utils import save_to_pickle, load_from_pickle, save_to_matlab
from src.utils.dsp_utils import normalize

# which dataset?
dataset_id = '011111'
L = 19556
c = constants['speed_of_sound']
Fs = constants['Fs']

# which microphonese?
mics_idxs = [0, 5, 10, 15, 20, 25]
I = len(mics_idxs)
K = 7

# which source?
srcs_idxs = [3]
J = len(srcs_idxs)

dataset_dir = './data/dECHORATE/'
path_to_processed = './data/processed/'
path_to_note_csv = dataset_dir + 'annotations/dECHORATE_database.csv'
path_to_manual_annotation = './data/processed/rirs_manual_annotation/20200504_15h04_gui_annotation.pkl'

note_dict = load_from_pickle(path_to_manual_annotation)
dset = DechorateDataset(path_to_processed, path_to_note_csv)

rirs = np.zeros([L, I, J])
mics = np.zeros([3, I])
srcs = np.zeros([3, J])
toas = np.zeros([K, I, J])

for i, m in enumerate(mics_idxs):
    for j, s in enumerate(srcs_idxs):

        dset.set_dataset(dataset_id)
        dset.set_entry(m, s)
        mic, src = dset.get_mic_and_src_pos()
        mics[:, i] = mic
        srcs[:, j] = src

        _, rir = dset.get_rir()
        rirs[:, i, j] = rir

        # we just want toas here
        _, toa, _, _, _ = dset.get_ism_annotation(k_order=2, k_reflc=K)
        toas[:, i, j] = note_dict['toa'][:K, m, s, 0]
        toas[:, i, j] = toa
        toas[0,  i, j] += 0/Fs*np.random.randn(*toa[0].shape)
        toas[1:, i, j] += 0/Fs*np.random.randn(*toa[1:].shape)

        print(np.array([toas[:, i, j], toa]).T)


#TODO: make it with real reflection

## PREPARING FOR DOKMANIC:
dokdict = {
    'D' : distance.squareform(distance.pdist(mics.T)),
    'rir': rirs,
    'delay' : 0,
    'c' : dset.c,
    'fs' : dset.Fs,
    'repeat' : False,
    'mics' : mics,
    'src' : srcs,
    'T_direct' : toas[0, :, :],
    'T_reflct' : toas[1:, :, :],
}

save_to_matlab('./recipes/room_geometry_estimation/data_rooge.mat', dokdict)

# now you can just download and run Dokmanic code
