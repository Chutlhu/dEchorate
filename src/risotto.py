import numpy as np

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

    def estimate_rirs(self, rec, ref, method=None, windowing=True):
        self.rec = rec
        self.ref = ref
        self.air = self.deconvolution(rec, ref, method=method, windowing=windowing)
        return self.air.copy()

    def deconvolve(self, y, x, method=None, windowing=True):
        # match sizes
        y, x = make_same_length(y, x)
        # smooth beginning and the end with window
        if windowing:
            y = smooth_beginning_and_end(y)
            x = smooth_beginning_and_end(x)

        for _ in range(n_chirps):
            if method == 'vincent':
                h = vincent_deconvolve(y, x)   # H = Y/X
            if method == 'wiener':
                h = wiener_deconvolve(y, x)    # H=Syx/Sxx
            if method == 'olivier':
                h = olivier_deconvolve(y, x)   # xcorrelation

        if method == 'koldovsky':
            h = koldovsky_deconvolve(y, x) # Time Domain LS
        return h

    def get_rt60(self):
        raise NotImplementedError
        pass
