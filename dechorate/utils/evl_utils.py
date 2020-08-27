import numpy as np


def snr_dB(num, den, time_support=None):
    if time_support is None:
        num = np.var(num)
        den = np.var(den)
    else:
        num = np.var(num[time_support])
        den = np.var(den[time_support])
    return 10*(np.log10(num) - np.log10(den))
