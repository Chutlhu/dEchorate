import numpy as np

from src.dataset import DechorateDataset
from src.utils.file_utils import save_to_pickle, load_from_pickle


# which dataset?
dataset_id = '011111'

# which microphonese?
mics_idxs = [0, 5, 10, 15]

# which source?
srcs_idxs = [0, 1, 2, 3, 4]

dataset_dir = './data/dECHORATE/'
path_to_processed = './data/processed/'
path_to_note_csv = dataset_dir + 'annotations/dECHORATE_database.csv'

dset = DechorateDataset(path_to_processed, path_to_note_csv)
dset.set_dataset(dataset_id)
for t, h, m, s in dset.get_rirs_mics_and_srcs_iterator(mics_idxs, srcs_idxs):
    print(h.shape)
    print(m)
    print(s)