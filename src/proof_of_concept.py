import numpy as no
import soundfile as sf
import pyroomacoustics as pra
import matplotlib.pyplot as plt

from risotto.risotto import Risotto
from risotto import algorithms
from risotto.utils.dsp_utils import *
from risotto.deconvolution import *


# h <- [pyroomacoustics, realRIR]
# x <- [expsinechirp, white-noise, squarewav_random_freq]
#       windowing <- [on/off] : fade in / fade out [maybe 20 ms]
# bandpass to select frequencies with blackman / binary selection
# snr <- -10:30 dB
# rt60 <- 0: 1 seconds

# metrics
# thr for allclose
# rmse
# wasserstein-distance?!
# rt60 est
# cosine distance

path_to_data = './data/raw/'
path_to_x = path_to_data + '01-chirp_log_14kHz_iter3-consolidated.wav'
path_to_y = path_to_data + '02-array 1-consolidated.wav'

## load audio
x, fs_x = sf.read(path_to_x, always_2d=True)
y, fs_y = sf.read(path_to_y, always_2d=True)
assert fs_y == fs_x
Fs = fs_y

y = y - np.mean(y, axis=0)
x = x - np.mean(x, axis=0)
# # plot audio
# plt.subplot(211)
# plt.imshow(10*np.log10(np.abs(X)**2))
# plt.subplot(212)
# plt.imshow(10*np.log10(np.abs(Y)**2))
# plt.show()
# 1/0

sframe, eframe = 0*Fs, 11*Fs

Lh = len(y[sframe:eframe, 0])
freq_mask = np.zeros(Lh)
freq_mask[800:136260] = 1
freq_mask[Lh-136260:Lh-800] = 1

h = classic_deconvolution(
    y[sframe:eframe, 0],
    x[sframe:eframe, 0],
    freq_mask = freq_mask
)

plt.plot(np.arange(len(h))/Fs, h)
plt.show()
1/0
