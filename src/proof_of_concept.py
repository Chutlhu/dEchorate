import numpy as np
import scipy as sp
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


path_to_data = './data/raw/recordings/'
path_to_x = path_to_data + 'chirp_log_14kHz_iter3.wav'
path_to_y = path_to_data + '2019-12-26__17-45-32_full_refl.wav'

for path_to_y in [path_to_data + '2019-12-26__17-45-32_full_refl.wav',
                  path_to_data + '2019-12-26__17-53-36_wall_c_abs.wav']:
    ## load audio
    x, fs_x = sf.read(path_to_x, always_2d=True)
    y, fs_y = sf.read(path_to_y, always_2d=True)
    assert fs_y == fs_x
    Fs = fs_y
    N = x.shape[0]
    J = y.shape[1]

    ## chirp repetition
    n_chirps = 3
    sample_per_iter = N/n_chirps
    nfft = int(2*sample_per_iter)

    # BP - from 80 to 7900 Hz, linear phase
    freq_mask = bandpass_blackman(4001, 110, 7950, Fs)

    H = np.zeros([nfft, J, n_chirps], dtype=np.complex64)

    for c in range(n_chirps):
        ssample, esample = int(c*sample_per_iter), int((c+1)*sample_per_iter)
        for j in range(J):
            H[:, j, c] = classic_deconvolution(
                            y[ssample:esample, 0],
                            x[ssample:esample, 0],
                            nfft=nfft,
                            freq_mask=freq_mask,
                            return_freq=True
            )

    H = np.sum(H, axis=-1)
    h = np.real(np.fft.ifft(H, axis=0))

    plt.plot(envelope(h[:, 0]))

plt.show()
