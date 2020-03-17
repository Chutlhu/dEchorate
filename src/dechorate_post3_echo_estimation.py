import soundfile as sf
import os
import os.path as pat
import h5py
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize, minimize_scalar

from src.dataset import DechorateDataset, SyntheticDataset
from src.cadzow import cadzow_denoise, condat_denoise
from src.utils.dsp_utils import *
from src.utils.file_utils import save_to_pickle, load_from_pickle
from src import constants

dataset_dir = './data/dECHORATE/'
path_to_processed = './data/processed/'
path_to_note_csv = dataset_dir + 'annotations/dECHORATE_database.csv'

datasets=['000000', '010000', '011000', '011100', '011110', '011111',
          '001000', '000100', '000010', '000001']


def build_all_rirs_matrix_and_annotation(params):

    Fs = params['Fs']
    I = params['I']
    J = params['J']
    D = params['D']
    K = params['K']
    Lt = params['Lt']
    data = params['data']
    L = int(Fs*Lt)

    all_rirs = np.zeros([L, I, J, D])
    toa_sym = np.zeros([K, I, J, D])
    amp_sym = np.zeros_like(toa_sym)
    ord_sym = np.zeros_like(toa_sym)
    wal_sym = np.chararray([K, I, J, D])

    for i in range(0, I):
        for j in range(0, J):
            for d, dataset in enumerate(datasets):

                # get real data
                dset = DechorateDataset(path_to_processed, path_to_note_csv)
                dset.set_dataset(dataset)
                dset.set_entry(i, j)
                mic_pos, src_pos = dset.get_mic_and_src_pos()
                time, h = dset.get_rir()
                assert dset.Fs == Fs

                # get simulation annotation
                synth_dset = SyntheticDataset()
                synth_dset.set_room_size(constants['room_size'])
                synth_dset.set_dataset(dataset)
                synth_dset.set_c(343)
                synth_dset.set_k_order(3)
                synth_dset.set_k_reflc(K)
                synth_dset.set_mic(mic_pos[0], mic_pos[1], mic_pos[2])
                synth_dset.set_src(src_pos[0], src_pos[1], src_pos[2])
                # times, hs = synth_dset.get_rir()
                amp, tau, wall, order = synth_dset.get_note()

                toa_sym[:, i, j, d] = tau
                amp_sym[:, i, j, d] = amp
                wal_sym[:, i, j, d] = wall
                ord_sym[:, i, j, d] = order

                # plt.plot(np.abs(h[:L])**p + .5*d, label=dataset)
                # if d == 0:
                #     plt.stem(tau*dset.Fs, amp)
                #     for k in range(K):
                #         plt.text(tau[k]*dset.Fs, 0.25, r'$\tau_{%s}^{%d}$' % (wall[k][0], order[k]))

                all_rirs[:, i, j, d] = h[:L]

            # plt.legend()
            # plt.show()

            # ## DIRECT PATH-DECONVOLUTION
            # Fs = dset.Fs
            # rir = rirs[0]
            # dp_idx = np.argmax(rir)
            # L = int(0.2*Fs)
            # rir = rir[:L]
            # def f(x):
            #     x1 = int(x)
            #     dp = rir[dp_idx-x1:dp_idx+x1]
            #     dp_deconv = np.real(np.fft.ifft(np.fft.fft(rir, L) / np.fft.fft(dp, L)))[:L]
            #     cost = np.linalg.norm(dp_deconv**2, ord=1)
            #     return cost
            # res = minimize_scalar(f, bounds=[20, 200], method='Bounded')
            # print(res)
            # print(res.x)

    toa_note = {
        'toa': toa_sym,
        'amp': amp_sym,
        'wall': wal_sym,
        'order': ord_sym,
    }
    np.save('./data/tmp/all_rirs.npy', all_rirs)
    save_to_pickle('./data/tmp/toa_note.pkl', toa_note)
    return all_rirs

def direct_path_deconvolution(all_rirs, params):
    L, I, J, D = all_rirs.shape
    all_rirs_dconv = np.zeros_like(all_rirs)
    Fs = params['Fs']

    for i in range(I):
        for j in range(J):
            # assume that the anechoic data have a nice direct path
            anechoic_rir = all_rirs[:, i, j, 0]
            dp_idx = np.argmax(np.abs(anechoic_rir))

            # we consider as direct path everything in the interval peak [-1 ms, +3 ms]
            curr_dp_index = [dp_idx - int(0.001*Fs), dp_idx + int(0.003*Fs)]
            dp = anechoic_rir[curr_dp_index[0]:curr_dp_index[1]]

            for d in range(D):
                rir = all_rirs[:, i, j, d]
                dp_deconv = np.real(np.fft.ifft(np.fft.fft(rir, L) / np.fft.fft(dp, L)))
                # restore the direct path
                offset = int(0.001*Fs)
                # dp_deconv = np.concatenate([np.zeros(offset), dp_deconv])

                all_rirs_dconv[offset:, i, j, d] = dp_deconv[:-offset]
    return all_rirs_dconv


def plot_rir_skyline(rirs, dataset, toa_sym, toa_peak, params):
    K = params['K']
    L, I, J, D = rirs.shape
    IJ = I*J
    # flat the src-mic axis
    c = 0
    rirs_skyline = np.zeros([L, IJ, D])
    toa_sym_skyline = np.zeros([K, IJ, D])
    for j in range(J):
        for i in range(I):
            for d in range(D):
                rir = rirs[:, i, j, d]
                rirs_skyline[:, c, d] = np.abs(rir/np.max(np.abs(rir)))
                toa_sym_skyline[:, c, d] = toa_sym[:, i, j, d]
            c += 1

    for j in range(J):
        plt.axvline(j*params['I'], color='C7')

    # process the Skyline for visualization
    L = 2000
    rirs_skyline = np.clip(rirs_skyline[:L, :, dataset]**2, 0, 0.4)

    plt.imshow(rirs_skyline, extent=[0, I*J, 0, L], aspect='auto')
    for k in range(7):
        plt.scatter(np.arange(IJ)+0.5, L - toa_sym_skyline[k, :, dataset]*params['Fs'], c='C%d'%(k+1), alpha=.6, label='Pyroom DP %d' % k)

    plt.tight_layout()
    plt.legend()
    plt.savefig('./reports/figures/rir_dp_dconv_skyline_after_calibration.pdf')
    plt.show()


def plot_overlapped_rirs(rirs, toa_note, params):
    L, I, J, D = rirs.shape
    K, I, J, D = toa_note['toa'].shape
    Fs = params['Fs']

    for j in range(J):
        for i in range(I):
            plt.figure(figsize=(16, 9))

            for d in range(D):

                rir = rirs[:, i, j, d]

                # if d == 0:
                #     plt.plot(rir**2 + 0.2*d, alpha=.2, color='C1')
                rir_to_plot = (normalize(rir))**2
                rir_to_plot = np.clip(rir_to_plot, 0, 0.34)
                rir_to_plot = normalize(rir_to_plot)
                plt.plot(rir_to_plot + 0.2*d)

                # Print the dataset name
                wall_code_name = 'fcwsen'
                wall_code = [int(i) for i in list(datasets[d])]
                curr_walls = [wall_code_name[w]
                              for w, code in enumerate(wall_code) if code == 1]
                plt.text(50, 0.07 + 0.2*d, datasets[d])
                plt.text(50, 0.03 + 0.2*d, curr_walls)

            # plot the echo information
            for k in range(K):
                toa = toa_note['toa'][k, i, j, d]
                amp = toa_note['amp'][k, i, j, d]
                wall = toa_note['wall'][k, i, j, d]
                order = toa_note['order'][k, i, j, d]
                plt.axvline(x=int(toa*Fs), alpha=0.5)
                plt.text(toa*Fs, 0.025, r'$\tau_{%s}^{%d}$' % (wall.decode(), order), fontsize=12)
            plt.xlim([0, 2000])
            plt.title('RIRs dataset %s\nmic %d, src %d' % (datasets[d], i, j))
            plt.show()


def write_rir_and_note_as_file(rirs, toa_note, params):
    L, I, J, D = rirs.shape
    K, I, J, D = toa_note['toa'].shape
    Fs = params['Fs']

    os.system('mkdir -p ./data/processed/rirs_manual_annotation/')

    for j in range(J):
        print('Processing src', j)

        os.system('mkdir -p ./data/processed/rirs_manual_annotation/src_%d/' % j)

        for i in range(I):

            path = './data/processed/rirs_manual_annotation/src_%d/mic_%d/' % (j,i)
            os.system('mkdir -p ' + path)

            for d in range(D):

                rir = rirs[:, i, j, d]

                rir_to_plot = (normalize(rir))**2
                rir_to_plot = np.clip(rir_to_plot, 0, 0.34)
                rir_to_plot = normalize(rir_to_plot)



                # Print the dataset name
                wall_code_name = 'fcwsen'
                wall_code = [int(i) for i in list(datasets[d])]
                curr_walls = [wall_code_name[w]
                              for w, code in enumerate(wall_code) if code == 1]
                str_curr_wall = '_'.join(curr_walls)
                sf.write(path + '%s_%s_rir.wav' % (datasets[d], str_curr_wall), rir_to_plot, Fs)

            # plot the echo information
            path_to_curr_note = path + '%s_note.txt' % (datasets[d])
            text = ''
            for k in range(K):
                toa = toa_note['toa'][k, i, j, d]
                amp = toa_note['amp'][k, i, j, d]
                wall = toa_note['wall'][k, i, j, d]
                order = toa_note['order'][k, i, j, d]
                text += '%1.6f\t%1.6f\ttau_%s_%d\n' % (toa, toa, wall.decode(), order)

            with open(path_to_curr_note, "w") as text_file:
                text_file.write(text)


# # FFT domain
# nfft = L
# freqs = np.linspace(0, dset.Fs//2, nfft//2)
# freqs_idx = np.arange(0, nfft//2)
# assert len(freqs) == len(freqs_idx)


# fstart, fend = (0, 16000)
# istart = np.argmin(np.abs(freqs-fstart))
# iend = np.argmin(np.abs(freqs-fend))
# print(istart, iend)
# sub_freqs = np.arange(istart, iend, step=1)
# F = len(sub_freqs)
# # frequency_step = np.diff(freqs[sub_freqs])[0]
# frequency_step = (dset.Fs/2)/(L/2-1)
# print('Delta Freq', frequency_step)

# h = np.abs(rirs[R])
# H = np.fft.fft(h, n=nfft)
# plt.plot(np.abs(H))
# H = H[sub_freqs]
# plt.plot(np.abs(np.concatenate([np.zeros(istart), H])))
# plt.show()

# P = F//2
# assert F > 2*K+1

# toep_hs = []
# for i, h in enumerate(rirs):

#     H = np.fft.fft(h)

#     H = H[sub_freqs]

#     if denoising:
#         print('Cadzow Denoising')
#         print(H.shape, P)
#         Th_P = make_toepliz_as_in_mulan(H, P)
#         Th = condat_denoise(Th_P, K, thr_Cadzow=1e-7)
#         # Th = cadzow_denoise(Th, K, thr_Cadzow=1e-7)
#     else:
#         Th = make_toepliz_as_in_mulan(H, K+1)

#     assert Th.shape[1] == K+1
#     toep_hs.append(Th)

# Th = np.concatenate(toep_hs, axis=0).squeeze()
# # Th = toep_hs[0]

# U, Sigma, Vh = np.linalg.svd(Th, full_matrices=False)
# a = np.conj(Vh[-1,:K+1]).squeeze()  # find it in the nullspace

# assert np.allclose(np.linalg.norm(a), 1, atol=1e-3)
# assert len(a) == K+1

# roots = np.roots(a)
# print('Est a')
# print(a.T)

# roots_ref = np.exp(-1j*2*np.pi*frequency_step * tau)
# a_ref = np.poly(roots_ref[::-1])
# print('Ref a')
# print(a_ref.T)

# # print('Annihilation with est', np.linalg.norm(Th @ a.reshape([K+1,1])))
# # print('Annihilation with ref', np.linalg.norm(Th @ a_ref.reshape([K+1, 1])))

# print(1/frequency_step)
# print(Lt)

# tau_est_mod_freq = np.sort(np.mod(np.angle(roots)/(-2*np.pi*frequency_step), 1/frequency_step))

# print('relative peaks ref', tau)
# print('relative peaks mod freq', tau_est_mod_freq)
# print('diff', np.abs(tau - tau_est_mod_freq)*dset.Fs)

# # for rir in rirs:
# # rir = rirs[R]
# for r, rir in enumerate(rirs):
#     # rir = np.abs(rir / np.max(np.abs(rir)))
#     plt.plot(np.arange(len(rir))/dset.Fs, rir**2, label='RIR %d' % r)
# plt.stem(tau, .5*np.ones_like(tau), use_line_collection = True,
#          linefmt='C3-', markerfmt='C3o', label='simulation peaks')
# plt.stem(tau_est_mod_freq, .5*np.ones_like(tau_est_mod_freq), use_line_collection=True,
#          linefmt='C4-', markerfmt='C4x', label='recovered peaks mod freq')
# plt.legend()
# plt.show()


if __name__ == "__main__":
    params = {
        'Fs' : 48000,
        'I' : 30,
        'J' : 4,
        'D' : len(datasets),
        'R' : 0,
        'K' : 15,
        'Lt' : 0.4,
        'data' : ['real', 'synth'][0]
    }

    ## BUILD ALL-RIRs MATRIX
    # ATTENTION: this data are the results of the direct path deconvolution
    # all_rirs = build_all_rirs_matrix_and_annotation(params)

    ## LOAD BACK THE DATA
    all_rirs = np.load('./data/tmp/all_rirs.npy')
    toa_note = load_from_pickle('./data/tmp/toa_note.pkl')

    ## DIRECT-PATH DECOVOLUTION
    all_rirs_clean = direct_path_deconvolution(all_rirs, params)
    np.save('./data/tmp/all_rirs_clean.npy', all_rirs_clean)

    ## RIR SKYLINE
    # plot_rir_skyline(all_rirs_clean, 5, toa_sym, None, params)

    ## PLOT OVERLAPPED RIRS
    plot_overlapped_rirs(all_rirs, toa_note, params)

    ## WRITE RIR as WAV to file for manual annotation with audacity
    # write_rir_and_note_as_file(all_rirs_clean, toa_note, params)

    denoising = True
    concatenate = False
    pass
