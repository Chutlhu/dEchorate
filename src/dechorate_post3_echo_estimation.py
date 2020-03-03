import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.dataset import DechorateDataset, SyntheticDataset
from src.utils.dsp_utils import make_toepliz_as_in_mulan
from src.utils.file_utils import save_to_pickle, load_from_pickle
from src import constants

dataset_dir = './data/dECHORATE/'
path_to_processed = './data/processed/'
path_to_note_csv = dataset_dir + 'annotations/dECHORATE_database.csv'

datasets = ('000000', '010000', '001000', '000100', '000010', '000001') #'011000', '011100', '011110')

i = 2
j = 2
data = ['real', 'synth'][1]
K = 15

rirs = []
for d in datasets:

    print(d)
    dset = DechorateDataset(path_to_processed, path_to_note_csv)

    dset.set_dataset(d)
    dset.set_entry(i, j)
    mic_pos, src_pos = dset.get_mic_and_src_pos()
    times, h = dset.get_rir()

    synth_dset = SyntheticDataset()
    synth_dset.set_room_size(constants['room_size'])
    synth_dset.set_dataset(d)
    synth_dset.set_c(343)
    synth_dset.set_mic(mic_pos[0], mic_pos[1], mic_pos[2])
    synth_dset.set_mic(src_pos[0], src_pos[1], src_pos[2])
    times, h = synth_dset.get_rir()
    amp, tau, _, _ = synth_dset.get_note()
    tau = tau[:K]
    assert len(tau) == K

    L = int(0.1*dset.Fs)
    rirs.append(h[:L])


save_to_pickle('./data/tmp/rirs.pkl', rirs)
rirs = load_from_pickle('./data/tmp/rirs.pkl')


# FFT domain
nfft = L
freqs = np.linspace(0, dset.Fs//2, nfft//2+1)
print(freqs)
freqs_idx = np.arange(0, nfft//2+1)
print(freqs_idx)
assert len(freqs) == len(freqs_idx)
F = len(freqs)

sub_freqs = np.arange(100, 1000, step=1)
frequency_step = np.diff(freqs[sub_freqs])[0]
print('Delta Freq', frequency_step)


toeps = []
for h in rirs:
    H = np.fft.rfft(h, n=nfft)

    H = H[sub_freqs]

    toep_h = make_toepliz_as_in_mulan(H, K+1)
    toeps.append(toep_h)


Th = np.concatenate(toeps, axis=0).squeeze()

Th = toeps[0]

def denoise(A, K, n_Cadzow=0):
    # run Cadzow denoising
    for cadzow_loop in range(n_Cadzow):
        # low-rank projection
        [U, s, V] = la.svd(A, full_matrices=False)
        s[-1] = 0
        A = U @ np.diag(s) @ V

        # enforce Toeplitz structure
        z = np.zeros((A.shape[0] + A.shape[1] - 1, 1), dtype=np.complex128)
        for i in range(z.shape[0]):
            z[i] = np.mean(np.diag(A, K - i))
        A = lam.toeplitz(z[K:], z[K::-1])
    return A

Th = denoise(Th, K, n_Cadzow=10)

U, Sigma, Vh = np.linalg.svd(Th, full_matrices=False)
a = np.conj(Vh[-1,:]).squeeze()  # find it in the nullspace

print(np.linalg.norm(a))
# assert np.linalg.norm(a) == 1
assert len(a) == K+1
print(Sigma)
print(np.linalg.norm(Th @ a))


roots = np.roots(a)
print('Est roots', np.abs(roots))
print('Est roots', np.angle(roots))
# roots = roots/np.abs(roots)
# a = np.poly(roots[::-1])
print('Est a')
print(a.T)

roots_ref = np.exp(-1j*2*np.pi*frequency_step * tau)
print('Ref roots', np.angle(roots_ref))
a_ref = np.poly(roots_ref[::-1])
print('Ref a')
print(a_ref.T)

print(np.linalg.norm(Th @ a.reshape([K+1,1])))
print(np.linalg.norm(Th @ a_ref.reshape([K+1, 1])))

time_loc = np.sort(np.angle(roots)/(-2*np.pi*frequency_step))
print('ref', time_loc - time_loc[0])
print('est', tau - tau[0])
# time_loc = time_loc - time_loc[0] + tau[0]

# print(time_loc)
# print(tau)

# # time_loc = time_loc + 536/dset.Fs

for rir in rirs:

    plt.plot(np.arange(len(rir))/dset.Fs, (rir/np.max(np.abs(rir)))**2)
    plt.stem(time_loc, np.ones_like(time_loc))

plt.show()
