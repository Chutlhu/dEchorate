import numpy as np
import matplotlib.pyplot as plt


def plt_time_signal(signal, fs, **kwargs):
    return plt.plot(np.arange(len(signal))/fs, signal, **kwargs)
