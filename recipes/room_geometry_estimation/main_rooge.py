import numpy as np
import scipy as sp
import peakutils as pk
import matplotlib.pyplot as plt

from scipy.spatial import distance

from src import constants
from src.dataset import DechorateDataset
from src.utils.mds_utils import trilateration
from src.utils.file_utils import save_to_pickle, load_from_pickle, save_to_matlab
from src.utils.dsp_utils import normalize, envelope

# which dataset?
dataset_id = '011111'
L = 19556
c = constants['speed_of_sound']
Fs = constants['Fs']

# which microphonese?
mics_idxs = [27]
I = len(mics_idxs)
K = 50

# which source?
srcs_idxs = [0, 1, 2, 3]
J = len(srcs_idxs)

dataset_dir = './data/dECHORATE/'
path_to_processed = './data/processed/'
path_to_note_csv = dataset_dir + 'annotations/dECHORATE_database.csv'
path_to_manual_annotation = './data/processed/rirs_manual_annotation/20200505_12h38_gui_annotation.pkl'

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
        toas[:7, i, j] = note_dict['toa'][:7, m, s, 0]


        rir = np.abs(normalize(rir))

        # base = envelope(np.abs(arir))
        # base = pk.envelope(base[:10000], deg=1024) - 0.01
        # peaks, _ = sp.signal.find_peaks(rir[:1000], height=(base[:1000], 1))
        # plt.plot(base, '--')

        if j == 3:
            rir = np.clip(rir, 0, 0.4)
            rir = normalize(rir)

        peaks, _ = sp.signal.find_peaks(rir[:int(0.08*Fs)], height=0.1, distance=10, prominence=0.2)

        plt.plot(np.abs(rir))
        # plt.plot(peaks, rir[peaks], "x")
        idx = np.argsort(rir[peaks])
        peaks = peaks[idx][:30]


        peaks = list(peaks) + [int(i*Fs) for i in toa[:7]]
        print(peaks)
        plt.plot(peaks, rir[peaks], "x")
        plt.show()


        k = len(peaks)
        toas[:k, i, j] = sorted(peaks)
        # plt.plot(np.abs(rirs[:, i, j]))
        # for k in range(K):
        #     plt.axvline(x=int( ))
        # plt.xlim(0, 1500)
        # plt.show()

        # toas[:, i, j] = toa
        # toas[0,  i, j] += 2/Fs*(np.random.random(*toa[0].shape) - 0.5)
        # toas[1:, i, j] += 10/Fs*(np.random.random(*toa[1:].shape) - 0.5)

        # tmp = np.array([toas[:, i, j], toa]).T
        # print(tmp)
        # print(np.abs(tmp[:, 0] - tmp[:, 1]))

# print(toas.squeeze())

# # we keep the notation: mics localizing sources.
# # so if we do the opposite with have to swap
if mics.shape[1] == 1 and srcs.shape[1] > 1:
    mics, srcs = srcs, mics


# # at this point echo labeling is already done manually
# # so we can do trilateration for retrieving the images position

# anchor_pos = mics
# distances = toas[0, :, :].squeeze()
# print(anchor_pos)
# print(distances)

# a, b = trilateration(anchor_pos, distances)
# print(a)
# print(b)
# 1/0



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
