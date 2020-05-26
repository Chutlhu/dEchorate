import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


from dechorate import constants
from dechorate.dataset import DechorateDataset, SyntheticDataset
from dechorate.utils.file_utils import save_to_pickle, load_from_pickle, save_to_matlab
from dechorate.utils.dsp_utils import normalize, envelope


dataset_dir = './data/dECHORATE/'
path_to_processed = './data/processed/'
path_to_note_csv = dataset_dir + 'annotations/dECHORATE_database.csv'
path_to_after_calibration = path_to_processed + \
    'post2_calibration/calib_output_mics_srcs_pos.pkl'

note_dict = load_from_pickle(path_to_after_calibration)
dset = DechorateDataset(path_to_processed, path_to_note_csv)
sdset = SyntheticDataset()


datasets = constants['datasets']
c = constants['speed_of_sound']
Fs = constants['Fs']
recording_offset = constants['recording_offset']

for i in range(I):
    for j in range(J):

rirs_real = np.zeros([L, I, J])
rirs_synt = np.zeros([L, I, J])
mics = np.zeros([3, I])
srcs = np.zeros([3, J])
toas = np.zeros([K, I, J])

for i, m in enumerate(mics_idxs):
    for j, s in enumerate(srcs_idxs):

        # positions from beamcon
        dset.set_dataset(dataset_id)
        dset.set_entry(m, s)
        mic, src = dset.get_mic_and_src_pos()
        mics[:, i] = mic
        srcs[:, j] = src
        _, rrir = dset.get_rir()


        # double check with synthetic data
        sdset = SyntheticDataset()
        sdset.set_room_size(constants['room_size'])
        sdset.set_dataset(dataset_id, absb=0.85, refl=0.15)
        sdset.set_c(c)
        sdset.set_k_order(17)
        sdset.set_mic(mics[0, i], mics[1, i], mics[2, i])
        sdset.set_src(srcs[0, j], srcs[1, j], srcs[2, j])
        _, srir = sdset.get_rir()


        # measure after calibration
        rirs_real[:, i, j] = normalize(rrir)
        rirs_synt[:, i, j] = normalize(srir)
