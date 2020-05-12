import numpy as np
import matplotlib.pyplot as plt

from dechorate import constants
from dechorate.dataset import DechorateDataset, SyntheticDataset
from dechorate.utils.file_utils import save_to_pickle, load_from_pickle, save_to_matlab
from dechorate.utils.dsp_utils import normalize

from blaster.blaster import Blaster
from blaster.sota import Sota_algos
from blaster.utils.dsp_utils import resample

# How many echoes?
K = 7

# which dataset?
dataset_id = '011111'
L = 19556
Fs = constants['Fs']
c = constants['speed_of_sound']

# which microphonese?
mics_idxs = [7, 12]
I = len(mics_idxs)

# which source?
srcs_idxs = [0]
J = len(srcs_idxs)

dataset_dir = './data/dECHORATE/'
path_to_processed = './data/processed/'
path_to_note_csv = dataset_dir + 'annotations/dECHORATE_database.csv'
path_to_after_calibration = path_to_processed + \
    'post2_calibration/calib_output_mics_srcs_pos.pkl'

note_dict = load_from_pickle(path_to_after_calibration)
dset = DechorateDataset(path_to_processed, path_to_note_csv)
sdset = SyntheticDataset()

rirs = np.zeros([L, I, J])
mics = np.zeros([3, I])
srcs = np.zeros([3, J])
toas = np.zeros([K, I, J])
amps = np.zeros([K, I, J])

for i, m in enumerate(mics_idxs):
    for j, s in enumerate(srcs_idxs):

        # positions from beamcon
        dset.set_dataset(dataset_id)
        dset.set_entry(m, s)
        mic, src = dset.get_mic_and_src_pos()
        mics[:, i] = mic
        srcs[:, j] = src
        _, rir = dset.get_rir()

        # measure after calibration
        # mics[:, i] = note_dict['mics'][:, m]
        # srcs[:, j] = note_dict['srcs'][:, s]
        rirs[:, i, j] = normalize(rir)

        # print(i, 'cali', mics[:, i] - np.array(constants['room_size']))

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
        amps[:7, i, j] = .6*np.ones_like(tau)


## USING BLASTER
duration = 0.5 # sec
targetFs = 16000

L1 = 4800
L = int(Fs*duration)
s1 = (np.random.rand(1*Fs) - 0.5) / 0.5

h1 = rirs[:, 0, 0]
h2 = rirs[:, 1, 0]

x1 = np.convolve(h1, s1, 'full')[L1:L1+L]
x2 = np.convolve(h2, s1, 'full')[L1:L1+L]

x1 = resample(x1, Fs, targetFs)
x2 = resample(x2, Fs, targetFs)

oldFs = Fs
Fs = targetFs

t_max = 0.050 # early reflection in 50 ms

# ## PREPARING FOR CROCCO:
# croccodict = {
#     'x1': x1,
#     'x2': x2,
#     'Fs': Fs,
#     'toas': toas,
#     't_max' : t_max,
# }

# save_to_matlab(
#     './recipes/acoustic_echo_retrieval/data_aer.mat', croccodict)

# ## RUN BSN
# init = {'h1': np.random.random(int(t_max*Fs)),
#         'h2': np.random.random(int(t_max*Fs))}
# algo = Sota_algos('bsn', t_max, Fs, 0.001, 100, False)
# h1, h2 = algo.estimate(x1, x2, init)

# print(h1)
# print(h2)
# 1/0

## RUN BLASTER
blaster = Blaster(x1, x2, t_max, Fs,
                  max_n_diracs=10, max_iter=200,
                  do_plot=False, domain='time', do_post_processing=True)
blaster.patience = 100
blaster.delta = 1e-5
h1_est, h2_est = blaster.run_with_lambda_path(starting_iter=20, step=5)

print(h1_est)
print(h2_est)
l = toas[0, 0, 0]
L = int(l * oldFs)

ax = plt.subplot(211)
plt.title('Channel 1')
plt.plot(np.arange(len(h1[L:]))/oldFs, normalize(np.abs(h1[L:])), label=r'$h_1$')
plt.stem(toas[:7, 0, 0] - l, amps[:7, 0, 0], linefmt='C0-' , markerfmt='C0x', basefmt='C0-' , label=r'$h_1$', use_line_collection=True)
plt.stem(h1_est.toa, h1_est.coeff, linefmt='C1-' , markerfmt='C1x', basefmt='C1-' , label=r'$\hat{h_1}$', use_line_collection=True)
plt.xlim([-0.001, 0.09])
plt.legend()

plt.subplot(212, sharex=ax)
plt.title('Channel 2')
plt.plot(np.arange(len(h2[L:]))/oldFs, normalize(np.abs(h2[L:])), label=r'$h_1$')
plt.stem(toas[:7, 1, 0] - l, amps[:7, 1, 0], linefmt='C0-' , markerfmt='C0x', basefmt='C0-' , label=r'$h_2$', use_line_collection=True)
plt.stem(h2_est.toa, h2_est.coeff, linefmt='C1-' , markerfmt='C1x', basefmt='C1-', label=r'$\hat{h_2}$', use_line_collection=True)
plt.xlim([-0.001, 0.09])
plt.legend()
plt.show()
