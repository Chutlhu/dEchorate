import numpy as np
import scipy as sp
import peakutils as pk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from scipy.spatial import distance

from src import constants
from src.dataset import DechorateDataset, SyntheticDataset
from src.utils.mds_utils import trilateration
from src.utils.file_utils import save_to_pickle, load_from_pickle, save_to_matlab
from src.utils.dsp_utils import normalize, envelope

# which dataset?
dataset_id = '011110'
L = 19556
c = constants['speed_of_sound']
Fs = constants['Fs']

# which microphonese?
# mics_idxs = [0, 1, 5, 6, 10, 11, 15, 16, 20, 21, 25]
mics_idxs = [0, 1, 5, 6, 10, 11, 15, 16, 20, 21, 25]
I = len(mics_idxs)
K = 50

# which source?
srcs_idxs = [0]
J = len(srcs_idxs)

dataset_dir = './data/dECHORATE/'
path_to_processed = './data/processed/'
path_to_note_csv = dataset_dir + 'annotations/dECHORATE_database.csv'
path_to_after_calibration = path_to_processed + 'post2_calibration/calib_output_mics_srcs_pos.pkl'

note_dict = load_from_pickle(path_to_after_calibration)
dset = DechorateDataset(path_to_processed, path_to_note_csv)
sdset = SyntheticDataset()

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
        # _, rir = dset.get_rir()

        # mics[:, i] = note_dict['mics'][:, m]
        # srcs[:, j] = note_dict['srcs'][:, s]
        # rirs[:, i, j] = rir

        # double check with synthetic data
        sdset = SyntheticDataset()
        sdset.set_room_size(constants['room_size'])
        sdset.set_dataset(dataset_id, absb=1, refl=0)
        sdset.set_c(c)
        sdset.set_k_order(1)
        sdset.set_k_reflc(7)
        sdset.set_mic(mics[0, i], mics[1, i], mics[2, i])
        sdset.set_src(srcs[0, j], srcs[1, j], srcs[2, j])
        amp, tau, wall, order, gen = sdset.get_note()

        toas[:7, i, j] = tau[:7]
        toas[:7, i, j] = note_dict['toa_sym'][:7, m, s]
        toas[:7, i, j] = note_dict['toa_pck'][:7, m, s]


print(srcs)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(mics[0, :], mics[1, :], mics[2, :], marker='o', label='mics init')
ax.scatter(srcs[0, :], srcs[1, :], srcs[2, :], marker='o', label='srcs init')

for k in range(7):
    d1 = toas[k, :, :] * c
    srcs_est, error = trilateration(mics.T, d1.squeeze())
    print(error)
    ax.scatter(srcs_est[0, :], srcs_est[1, :], srcs_est[2, :], marker='x', label='img %d' % k)


plt.legend()
plt.show()

1/0



distances = distance.squareform(distance.pdist(mics.T))
print(distances)

## PREPARING FOR DOKMANIC:
dokdict = {
    'D' : distances,
    'rirs': rirs,
    'delay' : 0,
    'c' : dset.c,
    'fs' : dset.Fs,
    'repeat' : False,
    'mics' : mics,
    'src' : srcs,
    'T_direct': toas[0, :, :].squeeze(),
    'T_reflct' : toas[1:, :, :].squeeze(),
}

save_to_matlab('./recipes/room_geometry_estimation/data_rooge.mat', dokdict)

# now you can just download and run Dokmanic code
