import pytest

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

from src.risotto import Risotto
from src.dsp_utils import *

np.random.seed(666)

def make_test_data(Fs, duration_x, duration_h, signal_kind, snr_dB=200):
    Fs = 16000
    Lx = int(duration_x * Fs)
    Lh = int(duration_h * Fs)


    if signal_kind == 'broadband':
        t = np.arange(int(Lx)/Fs)
        x = np.random.random(*t.shape)

    elif signal_kind == 'speech':
        i = np.random.randint(1, 14)
        x, Fs = sf.read('./data/raw/wallstreet/sp%d.wav' % (i))
        x = normalize_and_zero_mean(x[:Lx])
        t = np.arange(len(x))

    elif signal_kind == 'narrowband':
        Kx = 5
        frq = 4000*np.random.random(Kx)
        amp = np.random.random(Kx)
        x = np.zeros([Lx, Kx])
        t = np.arange(Lx)/Fs
        for i in range(Kx):
            x[:, i] = amp[i] * np.cos(2 * np.pi * frq[i] * t)
        x = np.sum(x, axis=1)

    Kh = 10
    h = np.zeros(Lh)
    loc = np.sort(np.random.randint(100, Lh-1, size=Kh))
    amp = np.sort(np.random.random(Kh))[::-1]
    for l, a in zip(loc, amp):
        h[l] = a

    # some zeros at the beginning
    delay = np.random.randint(1000)
    x = np.concatenate([np.zeros(delay), x, np.zeros(delay)])

    y = awgn(np.convolve(h, x), snr_dB)

    return y, h, x


def test_classic_narrowband_ongrid_delays_synchronized():
    Fs = 16000
    y, h, x = make_test_data(Fs, 1, 0.2, 'narrowband')

    y, x = make_same_length(y, x, 'max')
    h_est = classic_deconvolution(y, x)
    h_est, h =  make_same_length(h_est, h, 'max')

    assert np.allclose(h_est, h, atol=1e-5)


def test_classic_broadband_ongrid_delays_synchronized():
    Fs = 16000
    y, h, x = make_test_data(Fs, 1, 0.2, 'broadband')

    y, x = make_same_length(y, x, 'max')
    h_est = classic_deconvolution(y, x)
    h_est, h =  make_same_length(h_est, h, 'max')

    assert np.allclose(h_est, h, atol=1e-5)


def test_classic_speech_ongrid_delays_synchronized():
    Fs = 16000
    y, h, x = make_test_data(Fs, 1, 0.2, 'speech')

    y, x = make_same_length(y, x, 'max')
    h_est = classic_deconvolution(y, x)
    h_est, h =  make_same_length(h_est, h, 'max')

    assert np.allclose(h_est, h, atol=1e-4)


def test_wiener_narrowband_ongrid_delays_synchronized():
    Fs = 16000
    y, h, x = make_test_data(Fs, 1, 0.2, 'narrowband')

    y, x = make_same_length(y, x, 'max')

    h_est = wiener_deconvolution(y, x)

    h_est, h =  make_same_length(h_est, h, 'max')

    assert np.allclose(h_est, h, atol=1e-8)


def test_wiener_broadband_ongrid_delays_synchronized():
    Fs = 16000
    y, h, x = make_test_data(Fs, 1, 0.2, 'broadband')

    y, x = make_same_length(y, x, 'max')

    h_est = wiener_deconvolution(y, x)

    h_est, h =  make_same_length(h_est, h, 'max')

    assert np.allclose(h_est, h)


def test_wiener_speech_ongrid_delays_synchronized():
    Fs = 16000
    y, h, x = make_test_data(Fs, 1, 0.2, 'speech')

    y, x = make_same_length(y, x, 'max')

    h_est = wiener_deconvolution(y, x)

    h_est, h =  make_same_length(h_est, h, 'max')

    assert np.allclose(h_est, h, atol=1e-3)


def test_koldovsky_broadband_ongrid_delays_synchronized():
    Fs = 16000
    y, h, x = make_test_data(Fs, 1, 0.2, 'broadband', snr_dB=200)

    y, x = make_same_length(y, x)

    h_est = koldovsky_deconvolution(y, x, int(0.2*Fs), 0, 0)

    h_est, h =  make_same_length(h_est, h, 'max')

    assert np.allclose(h_est, h)


def test_koldovsky_narrowband_ongrid_delays_synchronized():
    Fs = 16000
    y, h, x = make_test_data(Fs, 1, 0.2, 'narrowband', snr_dB=200)

    y, x = make_same_length(y, x)

    h_est = koldovsky_deconvolution(y, x, int(0.2*Fs), 0, 0)

    h_est, h = make_same_length(h_est, h, 'max')

    assert np.allclose(h_est, h)


def test_koldovsky_speech_ongrid_delays_synchronized():
    Fs = 16000
    y, h, x = make_test_data(Fs, 1, 0.2, 'speech', snr_dB=200)

    y, x = make_same_length(y, x)

    h_est = koldovsky_deconvolution(y, x, int(0.2*Fs), 0, 0)

    h_est, h = make_same_length(h_est, h, 'max')

    assert np.allclose(h_est, h)


def test_koldovsky_broadband_synth_rir_synchronized():

