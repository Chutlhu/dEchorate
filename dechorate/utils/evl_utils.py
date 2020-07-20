import numpy as np


def snr_dB(num, den):
    num = np.var(num)
    den = np.var(den)
    return 10*(np.log10(num) - np.log10(den))