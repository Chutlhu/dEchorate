import zipfile
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from itertools import permutations

from dechorate import constants
from dechorate.dataset import DechorateDataset, SyntheticDataset
from dechorate.utils.file_utils import *
from dechorate.utils.dsp_utils import normalize, resample

from blaster.blaster import Blaster
from blaster.channel import Channel, anchor_hp
from blaster.utils.dsp_utils import stft

from mir_eval.onset import evaluate


curr_dir = './recipes/acoustic_echo_retrieval/'

exp = 'synt'

dataset_dir = './data/dECHORATE/'
path_to_processed = './data/processed/'
path_to_raw_zips = dataset_dir + '/recordings/room-%s.zip'
path_to_note_csv = dataset_dir + 'annotations/dECHORATE_database.csv'
path_to_after_calibration = path_to_processed + 'post2_calibration/calib_output_mics_srcs_pos.pkl'

path_to_interim = curr_dir + 'data/interim/' + exp + '/'
path_to_results = curr_dir + 'results/' + exp + '/'

make_dirs(path_to_interim)
make_dirs(path_to_results)

# Annotation, RIRs from measurements, 'equivalent' synthetic RIRs
note_dict = load_from_pickle(path_to_after_calibration)
rdset = DechorateDataset(path_to_processed, path_to_note_csv)
sdset = SyntheticDataset()

target_idxs = [0, 1, 2, 3]
dataset_idxs = [2, 4, 5]

for t in target_idxs:
    for d in dataset_idxs:


        ###############################################################################
        ##      INPUTS
        #
        arr_idx = 2
        data_kind = 1
        target_idx = t
        k_to_rake = 7

        print('arr_idx', arr_idx)
        print('data_kind', data_kind)
        print('target_idx', target_idx)
        print('rake', k_to_rake)

        # Some constant of the dataset
        L = constants['rir_length']
        Fs = constants['Fs']
        c = constants['speed_of_sound']
        L = constants['rir_length']

        # which dataset?
        dset = constants['datasets'][d]
        print(':: Dataset code', dset)
        print(dset)

        # which array?
        print(':: Array', arr_idx)
        mics_idxs = [(5*arr_idx + i) for i in range(5)]

        mics_idxs = [0, 5, 10, 15, 20, 25]
        print(':: Mics index', mics_idxs)
        I = len(mics_idxs)

        # which source?
        srcs_idxs = [target_idx]
        s = srcs_idxs[0]
        J = len(srcs_idxs)
        print(':: Srcs index', srcs_idxs)


        ###########################################################################
        ##   GET RAW FILES
        #

        path_to_curr_zip = path_to_raw_zips % dset
        archive = zipfile.ZipFile(path_to_curr_zip, 'r')

        # get rir from the recondings
        rdset.set_dataset(dset)
        rdset.set_entry(15, s)
        mic_pos, src_pos = rdset.get_mic_and_src_pos()
        rrir = rdset.get_rir(Fs=Fs)
        filename = list(rdset.get_file_database_id('noise'))[0]

        wav_bytes = archive.read('room-%s/%s.wav' % (dset, filename))
        tmp_wavfile = '/tmp/asd.wav'
        with open(tmp_wavfile, 'wb+') as f:
            f.write(wav_bytes)
        f.close()
        wav, fs = sf.read(tmp_wavfile)

        obs_rec = wav[:, mics_idxs]

        ###########################################################################
        ##   GET FILTERS AND ANNOTATION
        #

        # how many echoes to rake?
        K = k_to_rake  # all the first 7
        # in which order?
        tk_ordering = 'strongest'  # earliest, strongest, order


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
                rrir = rdset.get_rir(Fs=Fs)


                # measure after calibration
                mics[:, i] = note_dict['mics'][:, m]
                srcs[:, j] = note_dict['srcs'][:, s]

                # get synthetic rir
                sdset = SyntheticDataset()
                sdset.set_room_size(constants['room_size'])
                sdset.set_dataset(dset, absb=0.7, refl=0.2)
                sdset.set_c(c)
                sdset.set_fs(Fs)
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

        plt.plot(rirs_real[:, 0, 0], alpha=0.2)
        plt.plot(rirs_synt[:, 0, 0], alpha=0.2)
        plt.scatter(toas_cmds[:, 0, 0]*Fs, amps_cmds[:, 0, 0])
        plt.scatter(toas_peak[:, 0, 0]*Fs, amps_cmds[:, 0, 0])

        L = 4*Fs
        src = np.random.randn(L)
        obs_synt = []
        # obs_real = []
        for i in range(I):
            # obs_real.append(np.convolve(rirs_real[:, i, 0], src, 'full')[int(Fs):int(2*Fs), None])
            obs_synt.append(np.convolve(rirs_synt[:, i, 0], src, 'full')[int(Fs):int(2*Fs), None])

        # obs_real = np.concatenate(obs_real, axis=1)
        obs_synt = np.concatenate(obs_synt, axis=1)
        obs_rec = obs_rec[int(4.5*Fs): int(5.5*Fs), :]

        croccodict = {
            'observation_rec': obs_rec,
            'observation_synt': obs_synt,
            # 'observation_real': obs_real,
            # 'rirs_real': rirs_real,
            # 'rirs_synt': rirs_synt,
            'dataset' : dset,
            'target' : t,
            'mics': mics,
            'srcs': srcs,
            'toas_synt': toas_cmds,
            'toas_peak': toas_peak,
            'amps_synt': amps_cmds,
            'Fs': Fs,
        }

        filename = 'data4crocco_dataset-%d_target-%d.mat' % (d, t)

        save_to_matlab(path_to_interim + filename, croccodict)


# ###############################################################################
# ##  ESTIMATION WITH BLASTER
# #
# obs = obs_synt
# rir = rirs_synt
# i1 = 0
# i2 = 2

# X2 = stft(obs[:, 0].T, Fs=Fs, nfft=2048, hop=1024)[-1]
# X1 = stft(obs[:, 2].T, Fs=Fs, nfft=2048, hop=1024)[-1]

# X2 = np.mean(X2, axis=1)
# X1 = np.mean(X1, axis=1)

# h1_ref = Channel(coeff=list(amps_cmds[:, i1, 0]), toa=list(toas_cmds[:, i1, 0]), fs=Fs)
# h2_ref = Channel(coeff=list(amps_cmds[:, i2, 0]), toa=list(toas_cmds[:, i2, 0]), fs=Fs)

# max_dist = 20
# t_max = max_dist/constants['speed_of_sound']

# ## RUN BLASTER
# blaster = Blaster(X1, X2, t_max, Fs, max_n_diracs=2,
#                   max_iter=20, domain='freq', do_post_processing=True)
# h1_est, h2_est = blaster.run_with_lambda_path(starting_iter=10, step=3)

# print('ref1:', np.round(np.array(h1_ref.toa)*Fs, 2))
# print('ref2:', np.round(np.array(h2_ref.toa)*Fs, 2))
# print('est1:', np.round(np.array(h1_est.toa)*Fs, 2))
# print('est2:', np.round(np.array(h2_est.toa)*Fs, 2))
