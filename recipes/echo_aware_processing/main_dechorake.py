import numpy as np
import matplotlib.pyplot as plt

from dechorate import constants
from dechorate.dataset import DechorateDataset, SyntheticDataset
from dechorate.utils.file_utils import save_to_pickle, load_from_pickle, save_to_matlab
from dechorate.utils.dsp_utils import normalize

from brioche.beamformer import DS, MVDR, LCMV

data_filename = './recipes/echo_aware_processing/data.pkl'

nfft = 1024
hop = 512
Fs = 16000

# beamformers
ds = DS(name='DS') # just direct path
dpmvdr = MVDR(name='dpmvdr', rcond=1e-17, fstart=200, fend=8000, Fs=Fs) # doa-based direct path
rkmvdr = MVDR(name='rkmvdr', rcond=1e-17, fstart=200, fend=8000, Fs=Fs) # rake
rtmvdr = MVDR(name='rtmvdr', rcond=1e-17, fstart=200, fend=8000, Fs=Fs) # relative
rlmvdr = MVDR(name='remvdr', rcond=1e-17, fstart=200, fend=8000, Fs=Fs) # relative early + late
rmsinr = MVDR(name='rmsinr', rcond=1e-17, fstart=200, fend=8000, Fs=Fs) # max-sinr
dplcmv = LCMV(name='dplcmv', rcond=1e-17, fstart=200, fend=8000, Fs=Fs) # doa-based direct path
rklcmv = LCMV(name='rklcmv', rcond=1e-17, fstart=200, fend=8000, Fs=Fs) # rake
rtlcmv = LCMV(name='rtlcmv', rcond=1e-17, fstart=200, fend=8000, Fs=Fs) # relative
relcmv = LCMV(name='relcmv', rcond=1e-17, fstart=200, fend=8000, Fs=Fs) # relative early + late

# Get the data: RIRs and Annotation
dataset_dir = './data/dECHORATE/'
path_to_processed = './data/processed/'
path_to_note_csv = dataset_dir + 'annotations/dECHORATE_database.csv'
path_to_after_calibration = path_to_processed + \
    'post2_calibration/calib_output_mics_srcs_pos.pkl'

note_dict = load_from_pickle(path_to_after_calibration)
rdset = DechorateDataset(path_to_processed, path_to_note_csv)
sdset = SyntheticDataset()

# Some constant of the dataset
L = constants['rir_length']
Fs = constants['Fs']
c = constants['speed_of_sound']
L = constants['rir_length']
datasets_name = constants['datasets'][:6]
D = len(datasets_name)

# which microphones?
mics_idxs = [15, 16, 17, 18, 19]
I = len(mics_idxs)
# which mic is the reference?
ref_mic = 3 # the third

# which source?
srcs_idxs = [0, 1]
J = len(srcs_idxs)
# which src is the one to enhance?
tgt_src = 0 # the first

# how many echoes to rake?
K = 7
# in which order?
order = 'order' # earliest, strongest, order


# rirs_real = np.zeros([L, I, J, D])
# rirs_synt = np.zeros([L, I, J, D])
# mics = np.zeros([3, I])
# srcs = np.zeros([3, J])
# toas = np.zeros([K, I, J])
# toas_synt = np.zeros_like(toas)
# toas_peak = np.zeros_like(toas)

# for d, dset in enumerate(datasets_name):
#     for i, m in enumerate(mics_idxs):
#         for j, s in enumerate(srcs_idxs):

#             # get rir from the recondings
#             rdset.set_dataset(dset)
#             rdset.set_entry(i, j)
#             mic, src = rdset.get_mic_and_src_pos()
#             _, rrir = rdset.get_rir()

#             # measure after calibration
#             mics[:, i] = note_dict['mics'][:, m]
#             srcs[:, j] = note_dict['srcs'][:, s]

#             # get synthetic rir
#             sdset = SyntheticDataset()
#             sdset.set_room_size(constants['room_size'])
#             sdset.set_dataset(dset, absb=0.85, refl=0.15)
#             sdset.set_c(c)
#             sdset.set_k_order(17)
#             sdset.set_mic(mics[0, i], mics[1, i], mics[2, i])
#             sdset.set_src(srcs[0, j], srcs[1, j], srcs[2, j])
#             # amp, tau, wall, order, gen = sdset.get_note()

#             _, srir = sdset.get_rir()
#             Ls = len(srir)

#             # measure after calibration
#             rirs_real[:, i, j, d] = rrir[:L]
#             rirs_synt[:Ls, i, j, d] = srir[:Ls]

#             toas_synt[:K, i, j] = note_dict['toa_sym'][:K, m, s]
#             toas_peak[:K, i, j] = note_dict['toa_pck'][:K, m, s]

# print('done with the extraction')
# rirs_real = np.squeeze(rirs_real)
# rirs_synt = np.squeeze(rirs_synt)

# data = {
#     'rirs_real' : rirs_real,
#     'rirs_synt' : rirs_synt,
#     'mics' : mics,
#     'srcs' : srcs,
#     'toas' : toas,
#     'toas_synt' : toas_synt,
#     'toas_peak' : toas_peak,
# }


# save_to_pickle(data_filename, data)
data = load_from_pickle(data_filename)
print('Data Loaded')

rirs_real = data['rirs_real']
rirs_synt = data['rirs_synt']
mics = data['mics']
srcs = data['srcs']
toas = data['toas']
toas_synt = data['toas_synt']
toas_peak = data['toas_peak']

# VISUALIZE THE RIR and the ANNOTATION