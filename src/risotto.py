import numpy as np

from src.dsp_utils import *

'''
Room Impulse reSpOnse esTimaTOr

author Diego Di Carlo
'''

class Risotto():
    def __init__(self, Fs):
        self.I = 0
        self.J = 0

        self.air = None
        self.ATF = None

        self.rec = None
        self.ref = None

        self.Fs = Fs
        pass

    def generate_reference_exp_sine_chirp(self, n_chirps, Fs, silence):
        return

    def estimate_rirs(self, rec, ref, method=None, windowing=True):
        self.rec = rec
        self.ref = ref
        self.air = self.deconvolve(rec, ref, method=method)
        return self.air.copy()

    def deconvolve(self, y, x, method=None, n_chirps=1, windowing=True, domain='TD', expected_duration=1):
        # match sizes
        y, x = make_same_length(y, x)

        Lh = int(expected_duration*self.Fs)
        h_est = np.zeros([Lh, n_chirps])

        for i in range(n_chirps):
            if method == 'vincent':
                h = classic_deconvolution(y, x)   # H = Y/X
            if method == 'wiener':
                h = wiener_deconvolution(y, x)    # H=Syx/Sxx
            if method == 'koldovsky':
                h = koldovsky_deconvolution(y, x, Lh)  # Time Domain LS
            L = h.shape[0]
            h_est[:L, i] = h

        # average the estimation
        h_est = np.mean(h_est)

        return h_est.copy()

    def estimate_rt60(self):
        raise NotImplementedError
        pass
