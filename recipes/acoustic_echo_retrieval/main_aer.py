import numpy as np
import matplotlib.pyplot as plt

from dechorate import constants
from dechorate.dataset import DechorateDataset, SyntheticDataset
from dechorate.utils.file_utils import save_to_pickle, load_from_pickle, save_to_matlab
from dechorate.utils.dsp_utils import normalize

from blaster.blaster import Blaster
from blaster.sota import Sota_algos
from blaster.utils.dsp_utils import resample

# Get the data
dataset_dir = './data/dECHORATE/'
path_to_processed = './data/processed/'
path_to_note_csv = dataset_dir + 'annotations/dECHORATE_database.csv'
path_to_after_calibration = path_to_processed + 'post2_calibration/calib_output_mics_srcs_pos.pkl'

note_dict = load_from_pickle(path_to_after_calibration)
dset = DechorateDataset(path_to_processed, path_to_note_csv)
sdset = SyntheticDataset()


# which dataset?
dataset_id = '011100'
L = 19556
Fs = constants['Fs']
c = constants['speed_of_sound']
L = constants['rir_length']

# which microphonese?
mics_idxs = [0, 1, 2, 3, 4]
I = len(mics_idxs)

# which source?
srcs_idxs = [0]
J = len(srcs_idxs)

# how many echoes?
K = 7

rirs_real = np.zeros([L, I, J])
rirs_synt = np.zeros([L, I, J])
mics = np.zeros([3, I])
srcs = np.zeros([3, J])
toas = np.zeros([K, I, J])
amps = np.zeros([K, I, J])

for i, m in enumerate(mics_idxs):
    for j, s in enumerate(srcs_idxs):

        # get rir from the recondings
        dset.set_dataset(dataset_id)
        dset.set_entry(i, j)
        mic, src = dset.get_mic_and_src_pos()
        mics[:, m] = mic
        srcs[:, s] = src
        _, rrir = dset.get_rir()
        rrir = normalize(rrir)

        # get synthetic rir
        sdset = SyntheticDataset()
        sdset.set_room_size(constants['room_size'])
        sdset.set_dataset(dataset_id, absb=0.85, refl=0.15)
        sdset.set_c(c)
        sdset.set_k_order(17)
        sdset.set_mic(mics[0, m], mics[1, m], mics[2, m])
        sdset.set_src(srcs[0, s], srcs[1, s], srcs[2, s])
        _, srir = sdset.get_rir(normalize=True)
        Ls = len(srir)

        # measure after calibration
        rirs_real[:, i, j] = rrir[:L]
        rirs_synt[:Ls, i, j] = srir[:Ls]

        tk, ak = sdset.get_note(ak_normalize=True, tk_order='strongest')
        toas[:K, i, j] = tk[:K]
        amps[:K, i, j] = ak[:K]

print('done with the extraction')
rirs_real = np.squeeze(rirs_real)
rirs_synt = np.squeeze(rirs_synt)
print('Data loaded.')

## CHECK DATA
i, j = 0, 0

plt.plot(rirs_real[:, i])
plt.plot(rirs_synt[:, i])
plt.scatter(toas[:, i, j]*Fs, amps[:, i, j])

## direct path deconvolution
def direct_path_deconvolution(rir, tau, pa, pb, Fs):
    p = int(tau*Fs)
    a = int(p - pa)
    b = int(p + pb)
    dp = rir[a:b]
    L = len(rir)
    dp_deconv = np.real(np.fft.ifft(np.fft.fft(rir, L) / np.fft.fft(dp, L)))[:L-pa]
    # restore the direct path
    dp_deconv = np.concatenate([np.zeros(pa), dp_deconv])
    return dp_deconv
dp_rir = direct_path_deconvolution(rirs_real[:, i], toas[0, i, j], 100, 120, Fs)

plt.plot(dp_rir)
plt.show()

dp_rirs_real = np.zeros_like(rirs_real)
for i in range(I):
    dp_rirs_real[:, i] = direct_path_deconvolution(rirs_real[:, i], toas[0, i, 0], 100, 120, Fs)



## PREPARING FOR CROCCO:
croccodict = {
    'hr'   : rirs_real,
    'hs'   : rirs_synt,
    'hd'   : dp_rirs_real,
    'Fs'   : Fs,
    'toas' : toas,
    'amps' : amps,
}

save_to_matlab(
    './recipes/acoustic_echo_retrieval/data_aer.mat', croccodict)
1/0

## ACOUSTIC ECHO RETRIEVAL
targetFs = 16000    # Hz
target_duration = 1 # sec

L1 = 4800
L = int(targetFs * target_duration)
s1 = (np.random.rand(1*Fs) - 0.5) / 0.5

plt.plot(s)
plt.title('source')
plt.show()

for i in range(I):
    plt.plot(np.abs(rirs_real[:, i]))
    plt.plot(np.abs(rirs_synt[:, i]), ls='--')

plt.title('filters')
plt.show()

x_real = []
x_synt = []
for i in range(I):
    x = np.convolve(rirs_real[:, i], s1, 'full')[L1:L1+L]
    x_real.append(resample(np.convolve(rirs_real[:, i], s1, 'full')[L1:L1+L], Fs, targetFs))
    x_synt.append(resample(np.convolve(rirs_synt[:, i], s1, 'full')[L1:L1+L], Fs, targetFs))

# plt.plot(x_real[0])
# plt.title('observations')
# plt.show()

x_real = np.concatenate([x[:, None] for x in x_real], axis=-1)
x_synt = np.concatenate([x[:, None] for x in x_synt], axis=-1)

oldFs = Fs
Fs = targetFs

t_max = np.max(toas) # early reflection in 50 ms

## RUN BSN
# init = {'h1': np.random.random(int(t_max*Fs)),
#         'h2': np.random.random(int(t_max*Fs))}
# algo = Sota_algos('bsn', t_max, Fs, 0.001, 100, False)
# h1, h2 = algo.estimate(x_real[:, 0], x_real[:, 1], init)


# # print(h1)
# # print(h2)
# # 1/0

# ## RUN BLASTER
# x1 = x_synt[:, 0]
# x2 = x_synt[:, 1]
# blaster = Blaster(x1, x2, t_max, Fs,
#                   max_n_diracs=10, max_iter=200,
#                   do_plot=False, domain='time', do_post_processing=True)
# blaster.patience = 100
# blaster.delta = 1e-5
# h1_est, h2_est = blaster.run_with_lambda_path(starting_iter=20, step=5)

# print(h1_est)
# print(h2_est)
# # l = toas[0, 0, 0]
# # L = int(l * oldFs)

# # ax = plt.subplot(211)
# # plt.title('Channel 1')
# # plt.plot(np.arange(len(h1[L:]))/oldFs, normalize(np.abs(h1[L:])), label=r'$h_1$')
# # plt.stem(toas[:7, 0, 0] - l, amps[:7, 0, 0], linefmt='C0-' , markerfmt='C0x', basefmt='C0-' , label=r'$h_1$', use_line_collection=True)
# # plt.stem(h1_est.toa, h1_est.coeff, linefmt='C1-' , markerfmt='C1x', basefmt='C1-' , label=r'$\hat{h_1}$', use_line_collection=True)
# # plt.xlim([-0.001, 0.09])
# # plt.legend()

# # plt.subplot(212, sharex=ax)
# # plt.title('Channel 2')
# # plt.plot(np.arange(len(h2[L:]))/oldFs, normalize(np.abs(h2[L:])), label=r'$h_1$')
# # plt.stem(toas[:7, 1, 0] - l, amps[:7, 1, 0], linefmt='C0-' , markerfmt='C0x', basefmt='C0-' , label=r'$h_2$', use_line_collection=True)
# # plt.stem(h2_est.toa, h2_est.coeff, linefmt='C1-' , markerfmt='C1x', basefmt='C1-', label=r'$\hat{h_2}$', use_line_collection=True)
# # plt.xlim([-0.001, 0.09])
# # plt.legend()
# # plt.show()
