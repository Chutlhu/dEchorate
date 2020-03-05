import h5py
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

from src.dataset import DechorateDataset, SyntheticDataset
from src.cadzow import cadzow_denoise, condat_denoise
from src.utils.dsp_utils import *
from src.utils.file_utils import save_to_pickle, load_from_pickle
from src import constants

dataset_dir = './data/dECHORATE/'
path_to_processed = './data/processed/'
path_to_note_csv = dataset_dir + 'annotations/dECHORATE_database.csv'

# datasets = ('000000', '010000', '001000', '000100', '000010', '000001') #'011000', '011100', '011110')
datasets = ('010000', '001000', '000100', '000010', '000001', '011000', '011100', '011110')
# datasets = ['010000']
# datasets = ('010000', '001000', '000100', '000010',
#             '000001', '011000', '011100', '011110')

i = 5
j = 4
R = 0
data = ['real', 'synth'][0]
K = 3
Lt = 0.015
k_order = 1
SNR = 0.01 if data == 'synth' else 0
denoising = True
concatenate = False

rirs = []
for d in datasets:

    print(d)
    dset = DechorateDataset(path_to_processed, path_to_note_csv)

    dset.set_dataset(d)
    dset.set_entry(i, j)
    mic_pos, src_pos = dset.get_mic_and_src_pos()
    time, h = dset.get_rir()
    h = h / np.max(np.abs(h))

    synth_dset = SyntheticDataset()
    synth_dset.set_room_size(constants['room_size'])
    synth_dset.set_dataset(d)
    print(synth_dset.absorption)
    synth_dset.set_c(343)
    synth_dset.set_k_order(k_order)
    synth_dset.set_k_reflc(K)
    synth_dset.set_mic(mic_pos[0], mic_pos[1], mic_pos[2])
    synth_dset.set_src(src_pos[0], src_pos[1], src_pos[2])
    times, hs = synth_dset.get_rir()
    amp, tau, _, _ = synth_dset.get_note()

    L = int(Lt*dset.Fs)
    h = h/np.max(np.abs(h))
    # plt.plot(time[:L], h[:L]**2)
    # plt.stem(tau, amp)
    # plt.show()

    if data == 'real':
        rirs.append(h[:L])
    if data == 'synth':
        hs = hs + SNR*np.random.randn(*hs.shape)
        rirs.append(hs[:L])

# h = np.concatenate([h[:,None] for h in rirs], axis=1)
# h = np.mean(h, axis=1)
# plt.plot(h**2)
# plt.show()
# rirs = [h]

# FFT domain
nfft = L
freqs = np.linspace(0, dset.Fs//2, nfft//2)
freqs_idx = np.arange(0, nfft//2)
assert len(freqs) == len(freqs_idx)


fstart, fend = (1000, 8000)
istart = np.argmin(np.abs(freqs-fstart))
iend = np.argmin(np.abs(freqs-fend))
print(istart, iend)
sub_freqs = np.arange(istart, iend, step=1)
F = len(sub_freqs)
# frequency_step = np.diff(freqs[sub_freqs])[0]
frequency_step = (dset.Fs/2)/(L/2-1)
print('Delta Freq', frequency_step)

h = rirs[R]
H = np.fft.fft(h, n=nfft)
plt.plot(np.abs(H))
H = H[sub_freqs]
plt.plot(np.abs(H))
plt.show()

P = F//2
assert F > 2*K+1

toep_hs = []
for i, h in enumerate(rirs):

    H = np.fft.fft(h)

    H = H[sub_freqs]

    if denoising:
        print('Cadzow Denoising')
        print(H.shape, P)
        Th_P = make_toepliz_as_in_mulan(H, P)
        Th = condat_denoise(Th_P, K, thr_Cadzow=1e-7)
        # Th = cadzow_denoise(Th, K, thr_Cadzow=1e-7)
    else:
        Th = make_toepliz_as_in_mulan(H, K+1)

    assert Th.shape[1] == K+1
    toep_hs.append(Th)

Th = np.concatenate(toep_hs, axis=0).squeeze()
# Th = toep_hs[0]

U, Sigma, Vh = np.linalg.svd(Th, full_matrices=False)
a = np.conj(Vh[-1,:K+1]).squeeze()  # find it in the nullspace

assert np.allclose(np.linalg.norm(a), 1, atol=1e-3)
assert len(a) == K+1

roots = np.roots(a)
print('Est a')
print(a.T)

roots_ref = np.exp(-1j*2*np.pi*frequency_step * tau)
a_ref = np.poly(roots_ref[::-1])
print('Ref a')
print(a_ref.T)

# print('Annihilation with est', np.linalg.norm(Th @ a.reshape([K+1,1])))
# print('Annihilation with ref', np.linalg.norm(Th @ a_ref.reshape([K+1, 1])))

print(1/frequency_step)
print(Lt)

tau_est_mod_freq = np.sort(np.mod(np.angle(roots)/(-2*np.pi*frequency_step), 1/frequency_step))
# tau_est_mod_hlen = np.sort(np.mod(np.angle(roots)/(-2*np.pi*frequency_step), Lt))
# tau_est_abs = np.sort(np.abs(np.angle(roots)/(-2*np.pi*frequency_step)))
# tau_est_tau = np.abs(np.abs(np.angle(roots)/(-2*np.pi*frequency_step)) - tau[0]) + tau[0]
# tau_est = np.sort(np.angle(roots)/(-2*np.pi*frequency_step))
print('relative peaks ref', tau)
print('relative peaks mod freq', tau_est_mod_freq)
# print('relative peaks mod lenh', tau_est_mod_hlen)
# print('relative peaks abs', tau_est_abs)
# print('relative peaks tau', tau_est_tau)

# for rir in rirs:
# rir = rirs[R]
for r, rir in enumerate(rirs):
    rir = np.abs(rir / np.max(np.abs(rir)))
    plt.plot(np.arange(len(rir))/dset.Fs, rir, label='RIR %d' % r)
# plt.plot(np.arange(len(rir))/dset.Fs, envelope(rir), label='RIR')
plt.stem(tau, .5*np.ones_like(tau), use_line_collection = True,
         linefmt='C3-', markerfmt='C3o', label='simulation peaks')
# plt.stem(tau_est_mod_hlen, .5*np.ones_like(tau_est_mod_hlen), use_line_collection=True,
#          linefmt='C2-', markerfmt='C2s', label='recovered peaks mod hlen')
plt.stem(tau_est_mod_freq, .5*np.ones_like(tau_est_mod_freq), use_line_collection=True,
         linefmt='C4-', markerfmt='C4x', label='recovered peaks mod freq')
plt.legend()
plt.show()
