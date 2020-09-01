import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import soundfile as sf

from dechorate import constants
from dechorate.dataset import DechorateDataset, SyntheticDataset
from dechorate.utils.file_utils import save_to_pickle, load_from_pickle, save_to_matlab
from dechorate.utils.dsp_utils import normalize, envelope

path_to_output = 'recipes/dataset_analysis/'

dataset_dir = './data/dECHORATE/'
path_to_processed = './data/processed/'
path_to_note_csv = dataset_dir + 'annotations/dECHORATE_database.csv'
path_to_after_calibration = path_to_processed + 'post2_calibration/calib_output_mics_srcs_pos.pkl'

# Annotation, RIRs from measurements, 'equivalent' synthetic RIRs
note_dict = load_from_pickle(path_to_after_calibration)
rdset = DechorateDataset(path_to_processed, path_to_note_csv)
sdset = SyntheticDataset()

datasets = constants['datasets']
L = constants['rir_length']
c = constants['speed_of_sound']
Fs = constants['Fs']
recording_offset = constants['recording_offset']

# which dataset?
dset = '011110'
print(':: Dataset code', dset)
print(dset)

# which mic?
mics_idxs = [5, 10, 15, 20, 25, 29]
print(':: Mics index', mics_idxs)
I = len(mics_idxs)

# which src?
srcs_idxs = [0]
s = srcs_idxs[0]
J = len(srcs_idxs)

rirs_real = np.zeros([L, I, J])
rirs_synt = np.zeros([L, I, J])
rirs_real_no_dp = np.zeros([L, I, J])
mics = np.zeros([3, I])
srcs = np.zeros([3, J])
toas = np.zeros([7, I, J])

dp_extremes = np.array(
    [[327, 460],
     [314, 465],
     [313, 431],
     [210, 380],
     [150, 316],
     [ 60, 270]])


def dp_deconv(rir, dp_extreme):
    print(dp_extreme)
    # take the dp from... anechoic
    a = dp_extreme[0]
    b = dp_extreme[1]
    dp = rir[a:b]
    p = np.argmax(np.abs(dp))

    rir_no_dp = np.real(np.fft.ifft(np.fft.fft(rir, L) / np.fft.fft(dp, L)))[:L-p]
    # restore the direct path
    return np.concatenate([np.zeros(p), rir_no_dp])

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
        tk, ak = sdset.get_note(
            ak_normalize=False, tk_order='strongest')

        ak = ak / (4 * np.pi)

        _, srir = sdset.get_rir(normalize=False)
        srir = srir / (4 * np.pi)

        Ls = min(len(srir), L)
        Lr = min(len(rrir), L)

        # measure after calibration
        rirs_real[:Lr, i, j] = rrir[:Lr]
        rirs_synt[:Ls, i, j] = srir[:Ls]

        # ordering in note dict
        toas[:7, i, j] = note_dict['toa_pck'][:7, m, s]

        # direct path deconv
        dp_extreme = dp_extremes[i, :]
        rir_no_dp = dp_deconv(rrir[:Lr], dp_extreme)

        Ld = len(rir_no_dp)
        rirs_real_no_dp[:Ld, i, j] = rir_no_dp

        sf.write(path_to_output + 'rir_measured_mic-%d_src-%d_room-%s.wav' %  (m, s, dset), rrir, Fs)
        sf.write(path_to_output + 'rir_synthetic_mic-%d_src-%d_room-%s.wav' % (m, s, dset), srir, Fs)
        sf.write(path_to_output + 'rir_dp_deconv_mic-%d_src-%d_room-%s.wav' % (m, s, dset), rir_no_dp, Fs)

np.savetxt(path_to_output + 'echo_locations.txt', toas.squeeze())
