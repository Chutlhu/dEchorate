import numpy as np
from scipy import stats

from dechorate.utils.dsp_utils import envelope

import matplotlib.pyplot as plt

def rt60_with_sabine(room_sizes, absorption):
    x, y, z = room_sizes
    volumes = x*y*z
    # total_surfaces = 2*(x*y + x*z + y*z)

    surf_east = y*z
    surf_west = y*z
    surf_north = x*z
    surf_south = x*z
    surf_ceiling = x*y
    surf_floor = x*y

    abs_south = absorption['south']
    abs_east = absorption['east']
    abs_west = absorption['west']
    abs_north = absorption['north']
    abs_south = absorption['south']
    abs_ceiling = absorption['ceiling']
    abs_floor = absorption['floor']

    equivalent_absorption_surface = (surf_east * abs_east) \
        + (surf_west * abs_west) \
        + (surf_north * abs_north) \
        + (surf_south * abs_south) \
        + (surf_ceiling * abs_ceiling) \
        + (surf_floor * abs_floor)

    rt60 = 0.161*volumes/equivalent_absorption_surface
    # rt60 = -0.161*volumes/(total_surfaces * np.log(1 - surface_abs/total_surfaces))

    return rt60


def find_nearest_value(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def rt60_from_rirs(h, Fs, M, snr=45, do_shroeder=True, val_min=-90):
    # for Fs = 48kHz, M=5001
    # Schroeder's limit of integration td depends on noise
    Lh = len(h)
    times = np.arange(Lh)/Fs
    ## global rt60
    # 1. envelope with Hilbert's transform
    h_amp = envelope(h)
    h_amp_dB = 20*np.log10(h_amp/np.max(h_amp))
    val_min = find_nearest_value(h_amp_dB[int(0.100*Fs):], val_min)
    idx_min = np.where(h_amp_dB == val_min)[0][0]
    L = int(max(0.100*Fs, min(idx_min, Fs)))
    h_amp = h_amp[:L]

    # 2. smooth
    h_amp_smooth = np.convolve(h_amp, np.ones(M)/M, mode='valid')
    E = h_amp_smooth/np.max(h_amp_smooth)
    EdB = 20*np.log10(E)
    # 3. Schroeder Integration method
    if do_shroeder:
        # 3.1 Reverse integration points according to the SNR
        h_amp_dB_tmp = 20*np.log10(h_amp_smooth/np.max(h_amp_smooth))
        idx_max = np.argmax(h_amp_dB_tmp)
        val_max = h_amp_dB_tmp[idx_max]
        h_amp_dB_tmp_cropped = h_amp_dB_tmp[idx_max:]
        snr = 45 if snr > 45 else snr
        val_max_less_snr = val_max - snr
        val_max_less_snr = find_nearest_value(
            h_amp_dB_tmp_cropped, val_max_less_snr)
        idx_max_less_snr = np.where(h_amp_dB_tmp == val_max_less_snr)[0][0]
        td = idx_max_less_snr + idx_max

        # 3.2 Compute the Schroeder intergration
        h_amp_schroeder = h_amp_smooth
        idx = np.arange(len(h_amp_smooth[:td]))[::-1]
        h_amp_schroeder[idx] = (
            np.cumsum(h_amp_smooth[idx])/np.sum(h_amp_smooth[:td]))
        E = h_amp_schroeder/np.max(h_amp_schroeder)
        EdB = 20*np.log10(E)

    # 4. perform linear regression
    # find a value which is 35dB less than our max
    idx_max = np.argmax(EdB)
    EdB_cropped = EdB[idx_max:]
    val_max = EdB[idx_max]
    val_max_less_snr_10 = val_max - (snr - 10)
    val_max_less_snr_10 = find_nearest_value(
                            EdB_cropped, val_max_less_snr_10)
    idx_max_less_snr_10 = np.where(EdB == val_max_less_snr_10)[0][0]

    val_max_less_5 = val_max - 5
    val_max_less_5 = find_nearest_value(
                        EdB_cropped, val_max_less_5)
    idx_max_less_5 = np.where(EdB == val_max_less_5)[0][0]

    # slice arrays to from max to max-35dB to calculate a linear regression for it
    x = times[idx_max_less_5:idx_max_less_snr_10]
    y = EdB[idx_max_less_5:idx_max_less_snr_10]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)


    # compute the final value
    rt60 = -60/slope
    return rt60


def compute_mixing_time(rir):
    raise NotImplementedError


def ddr_from_rir(rir, d, Fs, before=120, after=120, max_duration=None):
    '''
    Refer to the ACE challenge paper
    before = after = 120 sample for FS = 48000
    '''

    if Fs != 48000:
        raise NotImplementedError

    D = np.sum(np.abs(rir[d-before:d+after])**2)
    R = np.sum(np.sum(rir[d+after:])**2)

    ddr = 10*np.log10(D / R)

    return ddr
