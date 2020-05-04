import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import distance

from src.dataset import DechorateDataset
from src.utils.file_utils import save_to_pickle, load_from_pickle, save_to_matlab


# which dataset?
dataset_id = '011111'
L = 19556

# which microphonese?
mics_idxs = [2, 7, 12, 17, 22, 27]
I = len(mics_idxs)

# which source?
srcs_idxs = [0]
J = len(srcs_idxs)

dataset_dir = './data/dECHORATE/'
path_to_processed = './data/processed/'
path_to_note_csv = dataset_dir + 'annotations/dECHORATE_database.csv'

dset = DechorateDataset(path_to_processed, path_to_note_csv)
dset.set_dataset(dataset_id)

rirs = np.zeros([L, I, J])
mics = np.zeros([3, I])
srcs = np.zeros([3, J])
toas = np.zeros([7, I, J])

for i, m in enumerate(mics_idxs):
    for j, s in enumerate(srcs_idxs):

        dset.set_entry(m, s)
        mic, src = dset.get_mic_and_src_pos()
        _, rir = dset.get_rir()
        # we just want toas here
        _, toa, _, _, _ = dset.get_ism_annotation(k_order=1, k_reflc=7)

        mics[:, i] = mic
        srcs[:, j] = src
        rirs[:, i, j] = rir
        toas[:, i, j] = toa


## PREPARING FOR DOKMANIC:
croccodict = {
    'rir': rirs,
    'Fs': dset.Fs,
    'toas': toas
}

save_to_matlab(
    './recipes/acoustic_echo_retrieval/data_aer.mat', croccodict)
# now you can just download and run Dokmanic code
