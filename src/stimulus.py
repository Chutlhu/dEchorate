import numpy as np
import scipy.signal as sg
import soundfile as sf

import matplotlib.pyplot as plt

from src.utils.file_utils import save_to_pickle, load_from_matlab
from src.utils.dsp_utils import *


# =============================================================
# Credits to PyRirTool
# https: // github.com/maj4e/pyrirtool/blob/master/stimulus.py
# =============================================================

class ProbeSignal():
    def __init__(self, kind='exp_sine_sweep', fs=48000):

        if kind not in ['exp_sine_sweep', 'hadamard_noise', 'white_noise']:
            raise NameError('Excitation type not implemented')

        self.kind = kind
        self.fs = fs

        self.n_repetitions = 0
        self.n_seconds = 0
        self.total_duration = 0

        self.ampl_ranges = []
        self.freq_ranges = []
        self.time_ranges = []

        self.signal = None
        self.invfilter = None

        self.w = load_from_matlab('./data/raw/bp_filt_blackman_4000.mat')['Num'].squeeze()


    def save(self, path_to_output):
        sf.write(path_to_output, self.signal, self.fs)


    # Generate the stimulus and set requred attributes
    def generate(self, n_seconds, amplitude, n_repetitions, silence_at_start, silence_at_end, sweeprange):
        if self.kind == 'exp_sine_sweep':
            return self._generate_exponential_sine_sweep(n_seconds, amplitude, sweeprange, silence_at_start, silence_at_end, n_repetitions)
        # if self.kind == 'white_noise':
        #     return self._generate_white_noise(n_seconds, amplitude, silence_at_start, silence_at_end, n_repetitions)
        # if self.kind == 'hadamard_noise':
        #     return self._generate_white_noise(n_seconds, amplitude, silence_at_start, silence_at_end, n_repetitions)

        return None

    def _generate_exponential_sine_sweep(self, n_seconds, amplitude, sweeprange, silence_at_start, silence_at_end, n_repetitions):
        fs = self.fs

        f1=np.max((sweeprange[0], 1)) # start of sweep in Hz.
        if sweeprange[1] == 0:
            # end of sweep in Hz. Sweep till Nyquist to avoid ringing
            f2 = int(fs/2)
        else:
            f2 = sweeprange[1]
        self.freq_ranges = [f1, f2]

        w1 = 2*np.pi*f1/fs     # start of sweep in rad/sample
        w2 = 2*np.pi*f2/fs     # end of sweep in rad/sample

        n_samples = n_seconds*fs
        sinsweep = np.zeros(shape=(n_samples, 1))
        taxis = np.arange(0, n_samples, 1)/(n_samples-1)

        # for exponential sine sweeping
        lw = np.log(w2/w1)
        sinsweep = amplitude * \
            np.sin(w1*(n_samples-1)/lw * (np.exp(taxis*lw)-1))
        self.ampl_ranges = [-amplitude, amplitude]

        # # Find the last zero crossing to avoid the need for fadeout
        # # Comment the whole block to remove this
        # k = np.flipud(sinsweep)
        # error = 1
        # counter = 0
        # while error > 0.001:
        #     error = np.abs(k[counter])
        #     counter = counter+1

        # k = k[counter::]
        # sinsweep_hat = np.flipud(k)
        # sinsweep = np.zeros(shape=(n_samples,))
        # sinsweep[0:sinsweep_hat.shape[0]] = sinsweep_hat

        # the convolutional inverse
        envelope = (w2/w1)**(-taxis)  # Holters2009, Eq.(9)
        invfilter = np.flipud(sinsweep)*envelope
        scaling = np.pi*n_samples * \
            (w1/w2-1)/(2*(w2-w1)*np.log(w1/w2)) * \
            (w2-w1)/np.pi  # Holters2009, Eq.10
        invfilter = invfilter/amplitude**2/scaling

        # fade-in window. Fade out removed because causes ringing - cropping at zero cross instead
        taperStart = sg.tukey(n_samples, 1/16)
        taperWindow = np.ones(shape=(n_samples,))
        taperWindow[0:int(n_samples/2)] = taperStart[0:int(n_samples/2)]
        sinsweep = sinsweep*taperWindow

        taperEnding = sg.tukey(n_samples, 1/128)
        taperWindow = np.ones(shape=(n_samples,))
        taperWindow[int(n_samples/2):] = taperEnding[int(n_samples/2):]
        sinsweep = sinsweep*taperWindow

        # Final excitation including repetition and pauses
        sinsweep = np.expand_dims(sinsweep, axis=1)
        zerostart = np.zeros(shape=(silence_at_start*fs, 1))
        zeroend = np.zeros(shape=(silence_at_end*fs, 1))
        sinsweep = np.concatenate(
            (np.concatenate((zerostart, sinsweep), axis=0), zeroend), axis=0)
        sinsweep = np.transpose(
            np.tile(np.transpose(sinsweep), n_repetitions))

        times = np.arange(len(sinsweep))/fs

        # Set the attributes
        self.total_duration = fs*(silence_at_start + n_seconds + silence_at_end)
        self.invfilter = invfilter
        self.n_repetitions = n_repetitions
        self.signal = sinsweep
        self.times = times
        self.n_seconds = n_seconds
        self.n_repetitions = n_repetitions
        self.time_ranges = [silence_at_start, silence_at_end]

        return times.copy(), sinsweep.copy()


    def compute_rir(self, recording):

        if self.kind == 'exp_sine_sweep':

            assert len(recording.shape)  == 2
            I = recording.shape[1]
            Lr = self.total_duration

            nfft = 2*Lr
            Hinv = np.fft.fft(self.invfilter, n=nfft)
            W = np.fft.fft(self.w, n=nfft)
            t = len(self.invfilter)+2*self.fs
            Lh = 10*self.fs
            rirs = np.zeros(shape=(Lh, I))
            for i in range(I):

                x = recording[:3*Lr, i]
                x = x.reshape(3, Lr)

                X = np.fft.fft(x, n=nfft)
                H = np.mean(Hinv * X, 0)

                rirs[:, i] = np.real(np.fft.ifft(H * W))[t:t+Lh]

            return rirs

        else:
            raise NameError('Excitation type not implemented')


    def compute_delay(self, y, start=0, duration=10):
        s = int(start*self.fs)
        e = s + int(duration*self.fs)
        x = normalize(center(self.signal[s:e, :]))
        y = normalize(center(y[s:e, :]))

        # Cross-correlation of the two channels (same as convolution with one reversed)
        corr = sg.correlate(x, y, mode='same')

        # Find the offset of the peak. Zero delay would produce a peak at the midpoint
        delay = int(len(corr)/2) - np.argmax(corr)
        return delay


if __name__ == "__main__":
    Fs = 48000
    ps = ProbeSignal('exp_sine_sweep', Fs)
    n_seconds = 10
    amplitude = 0.3
    n_repetitions = 3
    silence_at_start = 2
    silence_at_end = 2
    sweeprange = [100, 14e3]
    t, s = ps.generate(n_seconds, amplitude, n_repetitions, silence_at_start, silence_at_end, sweeprange)
    sf.write('./data/processed/exp_sine_chirp_3rep_100-14kHz.wav', s, Fs)
    pass
