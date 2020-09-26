import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyroomacoustics as pra
import soundfile as sf

from tqdm import tqdm
from datetime import date

from dechorate import constants
from dechorate.dataset import DechorateDataset, SyntheticDataset
from dechorate.utils.file_utils import save_to_pickle, load_from_pickle, save_to_matlab, make_dirs
from dechorate.utils.dsp_utils import normalize, resample, envelope, todB, rake_filter
from dechorate.utils.viz_utils import plt_time_signal
from dechorate.utils.evl_utils import snr_dB

from risotto.rtf import estimate_rtf, estimates_PSDs_PSDr_from_RTF
from risotto.utils.dsp_utils import stft, istft

from brioche.beamformer import DS, MVDR, LCMV, GSC
from brioche.utils.dsp_utils import diffuse_noise

import speechmetrics
metrics = speechmetrics.load(['pesq', 'srmr'], 4)

curr_dir = './recipes/echo_aware_processing/'

exp = 'kowalczyk'
dataset_dir = './data/dECHORATE/'
path_to_processed = './data/processed/'
path_to_note_csv = dataset_dir + 'annotations/dECHORATE_database.csv'
path_to_after_calibration = path_to_processed + 'post2_calibration/calib_output_mics_srcs_pos.pkl'
path_to_interim  = curr_dir + 'data/interim/' + exp + '/'
path_to_results  = curr_dir + 'data/results/' + exp + '/'
path_to_wav_results = path_to_results + 'wav/'
make_dirs(path_to_interim)
make_dirs(path_to_results)
make_dirs(path_to_wav_results)

# Annotation, RIRs from measurements, 'equivalent' synthetic RIRs
note_dict = load_from_pickle(path_to_after_calibration)
rdset = DechorateDataset(path_to_processed, path_to_note_csv)
sdset = SyntheticDataset()
note_dict.keys()


def main(arr_idx, dataset_idx, target_idx, snr, data_kind, k_to_rake, spk_idx, ref_mic=0, render=False):

    print('arr_idx', arr_idx)
    print('data_kind', data_kind)
    print('dataset_idx', dataset_idx)
    print('target_idx', target_idx)
    print('snr', snr)
    print('rake', k_to_rake)
    print('spk', spk_idx)

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
    r = ref_mic
    print(':: Ref mics', mics_idxs[ref_mic])

    # which source?
    srcs_idxs = [target_idx]
    J = len(srcs_idxs)
    print(':: Srcs index', srcs_idxs)

    ###########################################################################
    ##   MAKE SOURCE
    #

    data_dir = curr_dir + 'TIMIT_long_nili/'
    s1m = data_dir + 'DR5_MHMG0_SX195_SX285_SX375_7s.wav'
    s2m = data_dir + 'DR7_MGAR0_SX312_SX402_7s.wav'
    s1f = data_dir + 'DR1_FTBR0_SX201_SI921_7s.wav'
    s2f = data_dir + 'DR4_FKLC0_SI985_SI2245_7s.wav'

    files = [s1m, s2m, s1f, s2f]
    src, fs = sf.read(files[spk_idx])
    Fs = fs
    print('Audio rate is', fs)
    print('Audio duration is', len(src)/fs)
    # center and scale for unit variance
    src = ((src-np.mean(src))/np.std(src))[:, None]
    assert len(src.shape) == 2
    assert src.shape[-1] == 1
    assert J == 1

    ###########################################################################
    ##   GET FILTERS AND ANNOTATION
    #

    # how many echoes to rake?
    K = k_to_rake  # all the first 7
    # in which order?
    tk_ordering = 'strongest'  # earliest, strongest, order

    # get tdoa
    dset = datasets_name[d]
    print(':: Dataset code', dset)

    rirs_real = np.zeros([L, I, J])
    rirs_synt = np.zeros([L, I, J])
    mics = np.zeros([3, I])
    srcs = np.zeros([3, J])
    toas = np.zeros([K, I, J])
    toas_cmds = np.zeros([K, I, J])
    amps_cmds = np.zeros([K, I, J])
    toas_peak = np.zeros([7, I, J])

    for i, m in enumerate(mics_idxs):
        for j, s in enumerate(srcs_idxs):

            # get rir from the recondings
            rdset.set_dataset(dset)
            rdset.set_entry(m, s)
            mic_pos, src_pos = rdset.get_mic_and_src_pos()
            rrir = rdset.get_rir(Fs=fs)
            # diffuse_noise = rdset.get_diffuse_noise(20)

            # measure after calibration
            mics[:, i] = note_dict['mics'][:, m]
            srcs[:, j] = note_dict['srcs'][:, s]

            # get synthetic rir
            sdset = SyntheticDataset()
            sdset.set_room_size(constants['room_size'])
            sdset.set_dataset(dset, absb=0.7, refl=0.2)
            sdset.set_c(c)
            sdset.set_fs(fs)
            sdset.set_k_order(30)
            sdset.set_k_reflc(30**3)
            sdset.set_mic(mics[0, i], mics[1, i], mics[2, i])
            sdset.set_src(srcs[0, j], srcs[1, j], srcs[2, j])
            tk, ak = sdset.get_note(ak_normalize=False, tk_order=tk_ordering)

            ak = ak / (4 * np.pi)

            _, srir = sdset.get_rir(normalize=False)
            srir = srir / (4 * np.pi)

            Ls = min(len(srir), L)
            Lr = min(len(rrir), L)

            # measure after calibration
            rirs_real[:Lr, i, j] = rrir[:Lr]
            rirs_synt[:Ls, i, j] = srir[:Ls]

            # ordering in note dict
            toas_peak[:7, i, j] = note_dict['toa_pck'][:7, m, s]
            toas_cmds[:K, i, j] = tk[:K]
            amps_cmds[:K, i, j] = ak[:K]

    print('done with the extraction')

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
    distance = np.linalg.norm(tgt_pos - mic_pos[:, r])
    print('Distance ::', distance)

    ###########################################################################
    ##   MAKE FILTERS
    #

    if data_kind == 'synt':
        # which rirs?
        h_full = rirs_synt
        # with annotation?
        toas = toas_cmds
        amps = amps_cmds

    if data_kind == 'real':

        h_full = rirs_real

        amps = np.zeros_like(amps_cmds)
        toas = np.zeros_like(toas_cmds)

        # with annotation?
        tk = toas_peak
        # restore amps based on direct path eight
        ak = np.zeros_like(toas_peak)
        for j in range(J):
            for i in range(I):
                for k in range(7):
                    t = int(toas_peak[k, i, j]*Fs)
                    a = np.max(np.abs(rirs_real[t-10:t+10, i, j]))
                    ak[k, i, j] = a

                # order for location
                if tk_ordering == 'earliest':
                    indices = np.argsort(tk[:, i, j])
                elif tk_ordering == 'pra_order':
                    indices = [0,1,2,3,4,5,6]
                elif tk_ordering == 'strongest':
                    indices = np.argsort(np.abs(ak[:, i, j]))[::-1]
                else:
                    raise ValueError('Wrong ordering option')


                amps[:, i, j] = ak[indices[:K], i, j]
                toas[:, i, j] = tk[indices[:K], i, j]

    # compute rt60
    rt60s = np.zeros([I, J])
    for i in range(I):
        for j in range(J):
            # plt.plot(normalize(rirs_synt[:, i, j]))
            try:
                rt60s[i, j] = pra.measure_rt60(h_full[:, i, j], fs=fs)
            except:
                rt60s[i, j] = np.nan
    rt60 = np.nanmean(rt60s.flatten())
    # tmix = np.floor(2e-3 * np.sqrt(room_size[]) * fs)


    h_early = h_full.copy()
    h_late = h_full.copy()
    tmix_ref = 0
    # divide the RIRs
    for i in range(I):
        for j in range(J):
            m = int(np.max(toas[:, i, j])*Fs+10)
            if i == ref_mic:
                tmix_ref_smpl = m
            h_late[:m + int(0.05*fs), i, j] = 0
            h_early[m:, i, j] = 0

            # plt.plot(np.abs(h_full[:, i, j]), alpha=0.7)
            # plt.plot(np.abs(h_early[:, i, j]), alpha=0.7)
            # plt.plot(np.abs(h_late[:, i, j]), alpha=0.7)
            # plt.scatter(toas[:, i,j ]*Fs, amps[:, i, j])
            # plt.axvline(x=m)
            # plt.show()

    h = {
        'full': h_full,
        'early': h_early,
        'late': h_late,
    }
    parts = h.keys()

    # Convolution, downsampling and stacking
    Lc = 10*Fs
    cs = {
        'full' : np.zeros([Lc, I, J]),
        'early' : np.zeros([Lc, I, J]),
        'late' : np.zeros([Lc, I, J]),
    }

    print('Convolution')
    for i in range(I):
        for j in range(J):
            for part in ['early', 'late']:
                print(part, i, j)
                c = np.convolve(h[part][:, i, j], src[:, j], 'full')
                L = len(c)
                cs[part][:L, i, j] = c[:L]

    cs['full'] = cs['early'] + cs['late']
    # save_to_pickle(curr_dir + 'cs_pkl', cs)
    # cs = load_from_pickle(curr_dir + 'cs_pkl')

    # sf.write(path_to_results + 'cs_full.wav', cs['full'][:, r, 0], fs)
    # sf.write(path_to_results + 'cs_late.wav', cs['late'][:, r, 0], fs)
    # sf.write(path_to_results + 'cs_early.wav', cs['early'][:, r, 0], fs)

    # Standardization wtr reference microphone
    sigma_target = np.std(cs['early'][:, r, 0])

    # lets add some silence in head and in tail
    for part in parts:
        # hereafter we assume that the two images have unit-variance at the reference microphone
        cs[part][:, :, 0] = cs[part][:, :, 0] / sigma_target
        cs[part] = np.concatenate([np.zeros([2*fs, I, J]), cs[part], np.zeros([2*fs, I, J])], axis=0)

    # diffuse noise field simulation given the array geometry
    dn_name = curr_dir + 'diffuse.npy'
    try:
        dn = np.load(dn_name)
    except:
        L = 20*fs
        dn = diffuse_noise(mic_pos, L, fs, c=343, N=32, mode='sphere').T
        np.save(dn_name, dn)

    dn = dn[:cs['full'].shape[0], :]

    # and unit-variance with respect to the ref mic
    dn = dn / np.std(dn[:, r])

    sigma_n = np.sqrt(10 ** (- snr / 10))

    cn = sigma_n * dn

    # mixing all together
    mix =  {
        'full': np.sum(cs['full'], axis=-1) + cn,
        'late': np.sum(cs['late'], axis=1) + cn,
        'early': np.sum(cs['early'], axis=1) + cn
    }

    vad = {
        'target': (int(2*fs), int(7*fs)),
        'noise':  (int(0.2*fs), int(1.8*fs)),
    }

    assert fs == 16000
    nfft = 1024
    hop = nfft // 2
    nrfft = nfft+1
    F = nrfft
    fstart = 350  # Hz
    fend = 7500  # Hz
    assert r == ref_mic


    # stft of the spatial images
    CSf = stft(cs['full'][:, :, 0].T, Fs=Fs, nfft=nfft, hop=hop)[-1]
    CSe = stft(cs['early'][:, :, 0].T, Fs=Fs, nfft=nfft, hop=hop)[-1]
    CSl = stft(cs['late'][:, :, 0].T, Fs=Fs, nfft=nfft, hop=hop)[-1]
    CN = stft(cn.T, Fs=Fs, nfft=nfft, hop=hop)[-1]
    X = stft(mix['full'].T, Fs=Fs, nfft=nfft, hop=hop)[-1]

    CSf = CSf.transpose([1, 2, 0])
    CSe = CSe.transpose([1, 2, 0])
    CSl = CSl.transpose([1, 2, 0])
    CN = CN.transpose([1, 2, 0])
    X = X.transpose([1, 2, 0])
    assert np.allclose(X, CSf+CN)

    x_in = istft(X[:, :, r], Fs=Fs, nfft=nfft, hop=hop)[-1].real
    csf_in = istft(CSf[:, :, r], Fs=Fs, nfft=nfft, hop=hop)[-1].real
    cse_in = istft(CSe[:, :, r], Fs=Fs, nfft=nfft, hop=hop)[-1].real
    csl_in = istft(CSl[:, :, r], Fs=Fs, nfft=nfft, hop=hop)[-1].real
    cn_in = istft(CN[:, :, r], Fs=Fs, nfft=nfft, hop=hop)[-1].real
    assert np.allclose(x_in, csf_in + cn_in)

    assert fs == 16000 == Fs
    freqs = np.linspace(0, fs//2, F)
    omegas = 2*np.pi*freqs

    print('full measured and synthetic RTF')
    gevdRTF = np.zeros([nrfft, I, J], dtype=np.complex)
    rakeRTF = np.zeros_like(gevdRTF)
    dpTF = np.zeros([F, I, J], dtype=np.complex)
    rkTF = np.zeros_like(gevdRTF)

    # mix with noise only
    xn = mix['full'][vad['noise'][0]:vad['noise'][1], :]
    # mix with target only
    xs = mix['full'][vad['target'][0]:vad['target'][1], :]

    for j, src in enumerate(['target']):

        Dr = rake_filter(np.ones(1), toas[:1, r, j], omegas)
        Hr = rake_filter(amps[:, r, j], toas[:, r, j], omegas)
        mr = xs[:, r]

        for i in range(I):

            if i == r:
                gevdRTF[:, r, j] = np.ones(nrfft, dtype=np.complex)
                rakeRTF[:, r, j] = np.ones(nrfft, dtype=np.complex)
                dpTF[:, r, j] = np.ones(nrfft, dtype=np.complex)

            else:
                # measured RTF
                mi = xs[:, i]
                nd = xn[:, [i,r]]
                gevdRTF[:, i, j] = estimate_rtf(mi, mr, 'gevdRTF', 'full', Lh=None, n=nd, Fs=fs, nfft=nfft, hop=hop)
                # early closed RTF
                Hi = rake_filter(amps[:, i, j], toas[:, i, j], omegas)
                rakeRTF[:, i, j] = Hi / Hr
                # direct path
                Di = rake_filter(np.ones(1), toas[:1, i, j], omegas)
                dpTF[:, i, j] = Di / Dr
                # Rake with far field
                rkTF[:, i, j] = Hi / Dr

    print('... done.')

    # computed Sigma_n from noise-only
    Sigma_n = np.zeros([F, I, I], dtype=np.complex64)
    N = stft(xn.T, Fs=Fs, nfft=nfft, hop=hop)[-1]
    N = N.transpose([1, 2, 0])
    for f in range(F):
        Sigma_n[f, :, :] = np.cov(N[f, :, :].T)
    print('Done with noise covariance.')

    # # compute early and late PSD
    # PSDs1, PSDr1, PSDl1, COVn = estimates_PSDs_PSDr_from_RTF(
    #     rakeRTF[:, :, 0],
    #     xs, xn,
    #     mic_pos, ref_mic = ref_mic,
    #     Fs=fs, nrfft=F, hop=hop, fstart=fstart, fend=fend, speed_of_sound=constants['speed_of_sound'])
    PSDl1 = np.zeros_like(Sigma_n)
    for f in range(F):
        PSDl1[f, :, :] = np.cov(CSl[f, :, :].T)

    Sigma_ln = np.zeros_like(Sigma_n)
    for f in range(F):
        Sigma_ln[f, :, :] = Sigma_n[f, :, :] + 0.5*PSDl1[f, :, :]

    bfs = [
        (DS(  name='dpDS', fstart=fstart, fend=fend, Fs=fs, nrfft=F).compute_weights(dpTF[:, :, 0]), dpTF),
        (MVDR(name='MVDR_dp', fstart=fstart, fend=fend, Fs=fs, nrfft=F).compute_weights(dpTF[:, :, 0], Sigma_n), dpTF),
        (MVDR(name='MVDR_rtf', fstart=fstart, fend=fend, Fs=fs, nrfft=F).compute_weights(gevdRTF[:, :, 0], Sigma_n), gevdRTF),
        (MVDR(name='MVDR_rtf_rake', fstart=fstart, fend=fend, Fs=fs, nrfft=F).compute_weights(rakeRTF[:, :, 0], Sigma_n), rakeRTF),
        (MVDR(name='MVDR_rake', fstart=fstart, fend=fend, Fs=fs, nrfft=F).compute_weights(rkTF[:, :, 0], Sigma_n), rkTF),
        (MVDR(name='MVDR_dp_late', fstart=fstart, fend=fend, Fs=fs, nrfft=F).compute_weights(dpTF[:, :, 0], Sigma_ln), dpTF),
        (MVDR(name='MVDR_rtf_late', fstart=fstart, fend=fend, Fs=fs, nrfft=F).compute_weights(gevdRTF[:, :, 0], Sigma_ln), gevdRTF),
        (MVDR(name='MVDR_rake_late', fstart=fstart, fend=fend, Fs=fs, nrfft=F).compute_weights(rakeRTF[:, :, 0], Sigma_ln), rakeRTF),
    ]

    results = []

    for (bf, RTF) in bfs:

        print('***', bf, '***')

        print('TARGET', np.mean(np.abs(bf.enhance(RTF[:, :, 0]))))

        # separation
        X_out = bf.enhance(X.copy())
        CSf_out = bf.enhance(CSf.copy())
        CSe_out = bf.enhance(CSe.copy())
        CSl_out = bf.enhance(CSl.copy())
        CN_out = bf.enhance(CN.copy())

        x_out = istft(X_out, Fs=fs, nfft=nfft, hop=hop)[-1].real
        csf_out = istft(CSf_out, Fs=fs, nfft=nfft, hop=hop)[-1].real
        cse_out = istft(CSe_out, Fs=fs, nfft=nfft, hop=hop)[-1].real
        csl_out = istft(CSl_out, Fs=fs, nfft=nfft, hop=hop)[-1].real
        cn_out = istft(CN_out, Fs=fs, nfft=nfft, hop=hop)[-1].real

        ref_in = cse_in
        ref_out = cse_out

        # metrics
        time = np.arange(vad['target'][0], vad['target'][1])
        snr_in = snr_dB(ref_in, cn_in, time)
        snr_out = snr_dB(cse_out, cn_out, time)
        print('SNR', snr_in, '-->', snr_out, ':', snr_out - snr_in)

        drr_in = snr_dB(ref_in, csf_in, time)
        drr_out = snr_dB(ref_out, csf_out, time)
        print('DRR', drr_in, '-->', drr_out, ':', drr_out - drr_in)

        snrr_in = snr_dB(ref_in, cn_in + csl_in, time)
        snrr_out = snr_dB(ref_out, cn_out + csl_out, time)
        print('SNRR', snrr_in, '-->', snrr_out, ':', snrr_out - snrr_in)

        pesq_in = metrics(x_in[time], ref_in[time], rate=fs)['pesq'][0]
        pesq_out = metrics(x_out[time], ref_in[time], rate=fs)['pesq'][0]
        print('PESQ', pesq_in, '-->', pesq_out, ':', pesq_out - pesq_in)

        srmr_in = metrics(x_in[time], ref_in[time], rate=fs)['srmr'][0]
        srmr_out = metrics(x_out[time], ref_in[time], rate=fs)['srmr'][0]
        print('SRMR', srmr_in, '-->', srmr_out, ':', srmr_out - srmr_in)


        suffix = '_data-%s_bf-%s' % (data_kind, bf)

        gin = np.abs(np.max(x_in))
        gout = np.abs(np.max(x_out))


        if render:
            sf.write(path_to_wav_results + 'x_out' + suffix + '.wav', x_out/gout, fs)
            sf.write(path_to_wav_results + 'cs_out' + suffix + '.wav', ref_out/gout, fs)
            sf.write(path_to_wav_results + 'x_in' + suffix + '.wav', x_in/gin, fs)
            sf.write(path_to_wav_results + 'cs_in' + suffix + '.wav', ref_in/gin, fs)


        result = {
            'bf' : str(bf),
            'snr_in': snr_in,
            'snr_out': snr_out,
            'snrr_in': snrr_in,
            'snrr_out': snrr_out,
            'pesq_in': pesq_in,
            'pesq_out': pesq_out,
            'srmr_in': srmr_in,
            'srmr_out': srmr_out,
            'rt60' : rt60,
            'distance' : distance,
            'drr_in' : drr_in,
            'drr_out' : drr_out,
        }
        results.append(result)

    return results



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Run Echo-aware Beamformers')
    parser.add_argument('-a', '--array', help='Which array?', required=True, type=int)
    parser.add_argument('-d', '--data', help='Real or Synthetic?', required=True, type=str)
    parser.add_argument('-D', '--dataset', help='Which dataset? from 0 to 6', required=True, type=int)
    parser.add_argument('-r', '--rake', help='how many echoes to rake', required=True, type=int)
    parser.add_argument('-N', '--snr', help='SNR input [dB]', required=True, type=int)
    parser.add_argument('-R', '--render', help='render?', required=False, type=bool, default=False)


    args = vars(parser.parse_args())

    data = args['data']
    dataset_idx = args['dataset']
    arr_idx = args['array']
    render = args['render']
    snr = args['snr']
    rake = args['rake']

    today = date.today()

    results = pd.DataFrame()

    # input('Data are %s\nWanna continue?' % data)

    suffix = 'arr-%d_data-%s_dataset-%d_snr-%d_rake-%d_bf-DSLate' % (arr_idx, data, dataset_idx, snr, rake)
    results.to_csv(path_to_results + '%s_results_%s.csv' % (today, suffix))

    c = 0
    for target_idx in range(4):
        for spk_idx in range(4):
            res = main(arr_idx, dataset_idx, target_idx, snr, data, rake, spk_idx, ref_mic=2, render=render)

            for res_bf in res:

                results.at[c, 'data'] = data
                results.at[c, 'array'] = arr_idx
                results.at[c, 'dataset'] = dataset_idx
                results.at[c, 'target_idx'] = target_idx
                results.at[c, 'spk_idx'] = spk_idx
                results.at[c, 'snr'] = snr
                results.at[c, 'bf'] = res_bf['bf']
                results.at[c, 'snr_in'] = res_bf['snr_in']
                results.at[c, 'snr_out'] = res_bf['snr_out']
                results.at[c, 'snrr_in'] = res_bf['snrr_in']
                results.at[c, 'snrr_out'] = res_bf['snrr_out']
                results.at[c, 'srmr_in'] = res_bf['srmr_in']
                results.at[c, 'srmr_out'] = res_bf['srmr_out']
                results.at[c, 'pesq_in'] = res_bf['pesq_in']
                results.at[c, 'pesq_out'] = res_bf['pesq_out']
                results.at[c, 'rt60'] = res_bf['rt60']
                results.at[c, 'distance'] = res_bf['distance']
                results.at[c, 'drr_in'] = res_bf['drr_in']
                results.at[c, 'drr_out'] = res_bf['drr_out']

                c += 1

        results.to_csv(path_to_results + '%s_results_%s.csv' % (today, suffix))
    pass
