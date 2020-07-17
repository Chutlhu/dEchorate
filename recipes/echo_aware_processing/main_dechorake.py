import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf

from dechorate import constants
from dechorate.dataset import DechorateDataset, SyntheticDataset
from dechorate.utils.file_utils import save_to_pickle, load_from_pickle, save_to_matlab
from dechorate.utils.dsp_utils import normalize, resample, envelope, todB, rake_filter
from dechorate.utils.viz_utils import plt_time_signal

from risotto.rtf import estimate_rtf
from risotto.utils.dsp_utils import stft, istft

from brioche.beamformer import DS, MVDR, LCMV, GSC
from brioche.utils.dsp_utils import diffuse_noise

import speechmetrics
metrics = speechmetrics.load('pesq', 4)

curr_dir = './recipes/echo_aware_processing/'
data_filename = curr_dir + 'data_notebook.pkl'

dataset_dir = './data/dECHORATE/'
path_to_processed = './data/processed/'
path_to_note_csv = dataset_dir + 'annotations/dECHORATE_database.csv'
path_to_after_calibration = path_to_processed + \
    'post2_calibration/calib_output_mics_srcs_pos.pkl'

# Annotation, RIRs from measurements, 'equivalent' synthetic RIRs
note_dict = load_from_pickle(path_to_after_calibration)
rdset = DechorateDataset(path_to_processed, path_to_note_csv)
sdset = SyntheticDataset()
note_dict.keys()


def main(arr_idx, dataset_idx, target_idx, interf_idx, sir, snr, data_kind):

    # Some constant of the dataset
    L = constants['rir_length']
    Fs = constants['Fs']
    c = constants['speed_of_sound']
    L = constants['rir_length']
    datasets_name = constants['datasets'][:6]
    D = len(datasets_name)

    # which dataset?
    d = dataset_idx

    # which array?
    print(':: Array', arr_idx)

    mics_idxs = [(5*arr_idx + i) for i in range(5)]
    print(':: Mics index', mics_idxs)
    I = len(mics_idxs)
    # which one is the reference mic?
    ref_mic = 4
    r = ref_mic
    print(':: Ref mics', mics_idxs[ref_mic])

    # which source?
    if interf_idx == target_idx:
        return []

    srcs_idxs = [target_idx, interf_idx]
    print(':: Srcs index', srcs_idxs)

    J = len(srcs_idxs)
    # which src is the one to enhance?
    t = 0  # the target
    q = 1  # the interf

    # how many echoes to rake?
    K = 7  # all the first 7
    # in which order?
    ordering = 'strongest'  # earliest, strongest, order

    # get tdoa
    dset = datasets_name[d]
    print(':: Dataset code', dset)

    rirs_real = np.zeros([L, I, J])
    rirs_synt = np.zeros([L, I, J])
    mics = np.zeros([3, I])
    srcs = np.zeros([3, J])
    toas = np.zeros([K, I, J])
    toas_peak = np.zeros([K, I, J])
    toas_cmds = np.zeros([K, I, J])
    amps_cmds = np.zeros([K, I, J])

    for i, m in enumerate(mics_idxs):
        for j, s in enumerate(srcs_idxs):

            # get rir from the recondings
            rdset.set_dataset(dset)
            rdset.set_entry(m, s)
            mic, src = rdset.get_mic_and_src_pos()
            _, rrir = rdset.get_rir()

            # measure after calibration
            mics[:, i] = note_dict['mics'][:, m]
            srcs[:, j] = note_dict['srcs'][:, s]

            # get synthetic rir
            sdset = SyntheticDataset()
            sdset.set_room_size(constants['room_size'])
            sdset.set_dataset(dset, absb=0.90, refl=0.2)
            sdset.set_c(c)
            sdset.set_fs(Fs)
            sdset.set_k_order(3)
            sdset.set_k_reflc(1000)
            sdset.set_mic(mics[0, i], mics[1, i], mics[2, i])
            sdset.set_src(srcs[0, j], srcs[1, j], srcs[2, j])
            tk, ak = sdset.get_note(ak_normalize=False, tk_order=ordering)

            ak = ak / (4 * np.pi)

            _, srir = sdset.get_rir(normalize=False)
            srir = srir / (4 * np.pi)

            Ls = min(len(srir), L)
            Lr = min(len(rrir), L)

            # measure after calibration
            rirs_real[:Lr, i, j] = rrir[:Lr]
            rirs_synt[:Ls, i, j] = srir[:Ls]

            toas_peak[:7, i, j] = note_dict['toa_pck'][:7, m, s]
            toas_cmds[:K, i, j] = tk[:K]
            amps_cmds[:K, i, j] = ak[:K]

    print('done with the extraction')
    rirs_real = np.squeeze(rirs_real)
    rirs_synt = np.squeeze(rirs_synt)

    data = {
        'rirs_real': rirs_real,
        'rirs_synt': rirs_synt,
        'mics': mics,
        'srcs': srcs,
        'toas_synt': toas_cmds,
        'toas_peak': toas_peak,
        'amps_synt': amps_cmds,
        'Fs': Fs,
    }
    mic_pos = mics
    arr_pos = np.mean(mics, 1)
    tgt_pos = srcs[:, 0:1]
    itf_pos = srcs[:, 1:2]

    save_to_pickle(data_filename, data)
    print('Saved.')

    data_dir = curr_dir + 'TIMIT_long_nili/'
    s1m = data_dir + 'DR5_MHMG0_SX195_SX285_SX375_7s.wav'
    s2m = data_dir + 'DR7_MGAR0_SX312_SX402_7s.wav'
    s1f = data_dir + 'DR1_FTBR0_SX201_SI921_7s.wav'
    s2f = data_dir + 'DR4_FKLC0_SI985_SI2245_7s.wav'

    files = [s1m, s2m, s1f, s2f]
    N = len(files)

    wavs = []
    for file in files:
        wav, fs = sf.read(file)
        assert len(wav.shape) == 1
        wavs.append(wav[:7*fs])
    print('done.')
    print('Audio rate is', fs)


    if data_kind == 'synt':
        # which rirs?
        h_ = rirs_synt
        # with annotation?
        toas = toas_cmds
        amps = amps_cmds
    if data_kind == 'real':
        h_ = rirs_real

        # restore amps based on direct path eight
        amps_rirs = np.zeros_like(toas_cmds)
        for j in range(J):
            for i in range(I):
                print(i, j)
                for k in range(K):
                    t = int(toas_peak[k, i, j]*Fs)
                    a = np.max(np.abs(rirs_real[t-5:t+5, i, j]))
                    amps_rirs[k, i, j] = a

        amps = amps_rirs
        toas = toas_peak


    r = ref_mic  # reference mic

    # inr = 15  # dB
    # snr = 15  # dB
    # sir = snr - inr  # dB

    print('Ref mic', r)
    print('input SIR', sir)
    print('input SNR', snr)

    s1 = wavs[0]
    s2 = wavs[2]

    # center and scale for unit variance
    ss1 = (s1-np.mean(s1))/np.std(s1)
    ss2 = (s2-np.mean(s2))/np.std(s2)
    assert len(ss1) == len(ss2)


    # Upsampling and stacking
    print('Upsampling for convolution', fs, '-->', Fs)
    s_ = np.concatenate([resample(ss1, fs, Fs)[:, None],
                        resample(ss2, fs, Fs)[:, None]], axis=1)
    print(s_.shape)

    Lc = 10*fs
    c_ = np.zeros([Lc, I, J])

    # Convolution, downsampling and stacking
    print('Convolution and downsampling', Fs, '-->', fs)
    for i in range(I):
        for j in range(J):
            cs = np.convolve(h_[:, i, j], s_[:, j], 'full')
            cs = resample(cs, Fs, fs)
            L = len(cs)
            print(i, j, L)
            c_[:L, i, j] = cs

    print('Done.')


    # Standardization wtr reference microphone
    sigma_target = np.std(c_[:7*fs, r, 0])
    sigma_interf = np.std(c_[:7*fs, r, 1])

    c_[:, :, 0] = c_[:, :, 0] / sigma_target
    c_[:, :, 1] = c_[:, :, 1] / sigma_interf

    # hereafter we assume that the two images have unit-variance at the reference microphone


    # lets add some silence and shift the source such that there is overlap
    cs1 = np.concatenate([np.zeros([2*fs, I]), c_[:, :, 0],
                        np.zeros([4*fs, I]), np.zeros([2*fs, I])], axis=0)
    cs2 = np.concatenate([np.zeros([2*fs, I]), np.zeros([4*fs, I]),
                        c_[:, :, 1], np.zeros([2*fs, I])], axis=0)
    # diffuse noise field simulation given the array geometry
    dn = diffuse_noise(mic_pos, cs1.shape[0], fs, c=343, N=32, mode='sphere').T
    assert dn.shape == cs1.shape
    # and unit-variance with respect to the ref mic
    dn = dn / np.std(dn[:, r])


    sigma_n = np.sqrt(10 ** (- snr / 10))
    sigma_i = np.sqrt(10 ** (- sir / 10))

    cs1 = cs1
    cdn = sigma_n * dn
    cs2 = sigma_i * cs2


    # mixing all together
    x = cs1 + cs2 + cdn

    vad = {
        'target': (int(2*fs), int(4.5*fs)),
        'interf': (int(10*fs), int(12.5*fs)),
        'noise': (int(0.5*fs), int(1.5*fs)),
    }

    x = cs1 + cs2 + cdn

    assert fs == 16000
    nfft = 1024
    hop = 512
    nrfft = nfft+1
    F = nrfft
    fstart = 200  # Hz
    fend = 5000  # Hz
    assert r == ref_mic

    # stft of the spatial images
    CS1 = stft(cs1.T, Fs=Fs, nfft=nfft, hop=hop)[-1]
    CS2 = stft(cs2.T, Fs=Fs, nfft=nfft, hop=hop)[-1]
    CDN = stft(cdn.T, Fs=Fs, nfft=nfft, hop=hop)[-1]
    X = stft(x.T, Fs=Fs, nfft=nfft, hop=hop)[-1]
    CS1 = CS1.transpose([1, 2, 0])
    CS2 = CS2.transpose([1, 2, 0])
    CDN = CDN.transpose([1, 2, 0])
    X = X.transpose([1, 2, 0])

    xin = istft(X[:, :, r], Fs=Fs, nfft=nfft, hop=hop)[-1].real
    cs1in = istft(CS1[:, :, r], Fs=Fs, nfft=nfft, hop=hop)[-1].real
    cs2in = istft(CS2[:, :, r], Fs=Fs, nfft=nfft, hop=hop)[-1].real
    cdnin = istft(CDN[:, :, r], Fs=Fs, nfft=nfft, hop=hop)[-1].real
    assert np.allclose(X, CS1+CS2+CDN)
    assert np.allclose(xin, cs1in + cs2in + cdnin)


    print('gved-based RTF.')
    dRTF = np.zeros([nrfft, I, J], dtype=np.complex)
    for j, src in enumerate(['target', 'interf']):
        for i in range(I):
            if i == r:
                dRTF[:, r, j] = np.ones(nrfft)
            else:
                print(vad[src][0], vad[src][1])
                mi = x[vad[src][0]:vad[src][1], i]
                mr = x[vad[src][0]:vad[src][1], r]
                nd = x[vad['noise'][0]:vad['noise'][1], [r, i]]
                dRTF[:, i, j] = estimate_rtf(mi, mr, 'gevdRTF', 'full', Lh=None, n=nd,
                                            Fs=fs, nfft=nfft, hop=hop)
    print('... done.')

    freqs = np.linspace(0, fs//2, F)
    omegas = 2*np.pi*freqs

    print('echo-based RTF:')
    eRTF = np.zeros([F, I, J], dtype=np.complex)
    for j in range(J):
        for i in range(I):
            if i == r:
                eRTF[:, r, j] = np.ones(nrfft)
            else:
                assert len(amps[:, i, j]) == K
                assert len(toas[:, i, j]) == K

                Hr = rake_filter(amps[:, r, j], toas[:, r, j], omegas)
                Hi = rake_filter(amps[:, i, j], toas[:, i, j], omegas)
                eRTF[:, i, j] = Hi / Hr
    print('... done.')


    print('direct-path-based RTF:')
    dpTF = np.zeros([F, I, J], dtype=np.complex)
    for j in range(J):
        for i in range(I):
            Hr = rake_filter(amps[:1, r, j], toas[:1, r, j], omegas)
            if i == r:
                dpTF[:, r, j] = np.ones(nrfft)
            else:
                Hi = rake_filter(amps[:1, i, j], toas[:1, i, j], omegas)
                dpTF[:, i, j] = Hi / Hr
    print('... done.')

    Sigma_n = np.zeros([F, I, I], dtype=np.complex64)
    for f in range(F):
        Sigma_n[f, :, :] = np.cov(CDN[f, :, :].T)
    print('Done with noise covariance.')

    bfs = [
        (DS(name='dpDS', fstart=fstart, fend=fend, Fs=fs, nrfft=F).compute_weights(dpTF[:, :, 0]), dRTF),
        (MVDR(name='rtfMVDR', fstart=fstart, fend=fend, Fs=fs, nrfft=F).compute_weights(dRTF[:, :, 0], Sigma_n), dRTF),
        (MVDR(name='ecoMVDR', fstart=fstart, fend=fend, Fs=fs, nrfft=F).compute_weights(eRTF[:, :, 0], Sigma_n), eRTF),
        (LCMV(name='rtfLCMV', fstart=fstart, fend=fend, Fs=fs, nrfft=F).compute_weights(dRTF, Sigma_n), dRTF),
        (LCMV(name='ecoLCMV', fstart=fstart, fend=fend, Fs=fs, nrfft=F).compute_weights(eRTF, Sigma_n), eRTF),
    ]

    results = []

    for (bf, RTF) in bfs:

        print('TARGET', np.mean(np.abs(bf.enhance(RTF[:, :, 0]))))
        print('INTERF', np.mean(np.abs(bf.enhance(RTF[:, :, 1]))))

        # separation
        Xout = bf.enhance(X.copy())
        CS1out = bf.enhance(CS1.copy())
        CS2out = bf.enhance(CS2.copy())
        CDNout = bf.enhance(CDN.copy())

        xout = istft(Xout, Fs=fs, nfft=nfft, hop=hop)[-1].real
        cs1out = istft(CS1out, Fs=fs, nfft=nfft, hop=hop)[-1].real
        cs2out = istft(CS2out, Fs=fs, nfft=nfft, hop=hop)[-1].real
        cdnout = istft(CDNout, Fs=fs, nfft=nfft, hop=hop)[-1].real


        # # plot
        # plt.figure(figsize=(16, 4))
        # plt.plot(xout, label='out')
        # plt.plot(xin, alpha=0.5, label='in')
        # plt.legend(loc='upper left')
        # plt.show()

        # plt.figure(figsize=(16, 4))
        # plt.subplot(131)
        # plt.plot(cs1out[2*fs:9*fs], label='cs1 out')
        # plt.plot(cs1in[2*fs:9*fs], alpha=0.5, label='cs1')
        # plt.legend(loc='upper left')
        # plt.subplot(132)
        # plt.plot(cs2out[6*fs:13*fs], label='cs2 out')
        # plt.plot(cs2in[6*fs:13*fs], alpha=0.5, label='cs2')
        # plt.legend(loc='upper left')
        # plt.subplot(133)
        # plt.plot(cdnout[2*fs:9*fs], label='cdn out')
        # plt.plot(cdnin[2*fs:9*fs], alpha=0.5, label='cdn')
        # plt.legend(loc='upper left')
        # plt.show()

        # metrics
        sar_out = todB(np.var(cs1in[2*fs:9*fs]) / np.var(cs1out[2*fs:9*fs]))
        print('SAR', sar_out)

        snr_in = todB(np.var(cs1in[2*fs:9*fs]) / np.var(cdnin[2*fs:9*fs]))
        snr_out = todB(np.var(cs1out[2*fs:9*fs]) / np.var(cdnout[2*fs:9*fs]))
        print('SNR', snr_in, '-->', snr_out, ':', snr_out - snr_in)

        sir_in = todB(np.var(cs1in[2*fs:9*fs]) / np.var(cs2in[6*fs:13*fs]))
        sir_out = todB(np.var(cs1out[2*fs:9*fs]) / np.var(cs2out[6*fs:13*fs]))
        print('SIR', sir_in, '-->', sir_out, ':', sir_out - sir_in)

        pesq_in = metrics(xin[7*fs:9*fs], cs1in[7*fs:9*fs], rate=fs)['pesq'][0]
        pesq_out = metrics(xout[7*fs:9*fs], cs1in[7*fs:9*fs], rate=fs)['pesq'][0]
        print('PESQ', pesq_in, '-->', pesq_out, ':', pesq_out - pesq_in)

        result = {
            'bf' : str(bf),
            'sar_out' : sar_out,
            'sir_in': sir_in,
            'snr_in': snr_in,
            'sir_out' : sir_out,
            'snr_out': snr_out,
            'pesq_in': pesq_in,
            'pesq_out': pesq_out,
        }
        results.append(result)

    return results

if __name__ == "__main__":

    data = 'real'

    results = pd.DataFrame()
    results.to_csv(curr_dir + 'results_%s.csv' % data)

    input('Data are %s\nWanna continue?' % data)

    target_idx = 0
    interf_idx = 2

    c = 0
    for arr_idx in range(5):
        for dataset_idx in range(5):
            for target_idx in range(4):
                for sir in [0, 10, 20]:
                    for snr in [0, 10, 20]:

                        try:
                            res = main(arr_idx, dataset_idx, target_idx, interf_idx, sir, snr, data)
                        except:
                            continue

                        if len(res) == 0:
                            continue

                        for res_bf in res:

                            results.at[c, 'array'] = arr_idx
                            results.at[c, 'dataset'] = dataset_idx
                            results.at[c, 'target_idx'] = target_idx
                            results.at[c, 'interf_idx'] = interf_idx
                            results.at[c, 'sir'] = sir
                            results.at[c, 'snr'] = snr
                            results.at[c, 'bf'] = res_bf['bf']
                            results.at[c, 'sar_out'] = res_bf['sar_out']
                            results.at[c, 'sir_in'] = res_bf['sir_in']
                            results.at[c, 'snr_in'] = res_bf['snr_in']
                            results.at[c, 'sir_out'] = res_bf['sir_out']
                            results.at[c, 'snr_out'] = res_bf['snr_out']
                            results.at[c, 'pesq_in'] = res_bf['pesq_in']
                            results.at[c, 'pesq_out'] = res_bf['pesq_out']

                            c += 1

                        results.to_csv(curr_dir + 'results.csv')
    pass
