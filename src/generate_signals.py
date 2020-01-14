import numpy as np
import soundfile as sf

# =============================================================
# Credits to PyRirTool
# https: // github.com/maj4e/pyrirtool/blob/master/stimulus.py
# =============================================================

class ProbeSignal():
    def __init__(self, kind='exp_sine_sweep', fs=48000):

        if kind not in ['exp_sine_sweep', 'hadamard_noise', 'white_noise']
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

    def save(self, path_to_output):
        sf.write(path_to_output, self.signal, self.fs)


    # Generate the stimulus and set requred attributes
    def generate(self, n_seconds, amplitude, n_repetitions, silence_at_start, silence_at_end, sweeprange):

        if self.kind == 'sinesweep':
            return self._generate_exponential_sine_sweep(fs, n_seconds, amplitude, sweeprange, silence_at_start, silence_at_end, n_repetitions)
        if self.kind == 'white_noise':
            return self._generate_white_noise(fs, n_seconds, amplitude, silence_at_start, silence_at_end, n_repetitions)
        if self.kind == 'hadamard_noise':
            return self._generate_white_noise(fs, n_seconds, amplitude, silence_at_start, silence_at_end, n_repetitions)

        return None


    def _generate_exponential_sine_sweep(self, n_seconds, amplitude, sweeprange, n_repetitions, silence_at_start, silence_at_end):
        fs = self.fs

        f1=np.max((sweeprange[0], 1)) # start of sweep in Hz.
        if sweeprange[1] == 0:
            # end of sweep in Hz. Sweep till Nyquist to avoid ringing
            f2 = int(fs/2)
        else:
            f2 = sweeprange[1]
        self.freq_ranges = [f1, f2]

        w1 = 2*pi*f1/fs     # start of sweep in rad/sample
        w2 = 2*pi*f2/fs     # end of sweep in rad/sample

        n_samples = n_seconds*fs
        sinsweep = np.zeros(shape=(n_samples, 1))
        taxis = np.arange(0, n_samples, 1)/(n_samples-1)

        # for exponential sine sweeping
        lw = log(w2/w1)
        sinsweep = amplitude * \
            sin(w1*(n_samples-1)/lw * (exp(taxis*lw)-1))
        self.ampl_ranges = [-amplitude, amplitude]

        # Find the last zero crossing to avoid the need for fadeout
        # Comment the whole block to remove this
        k = np.flipud(sinsweep)
        error = 1
        counter = 0
        while error > 0.001:
            error = np.abs(k[counter])
            counter = counter+1

        k = k[counter::]
        sinsweep_hat = np.flipud(k)
        sinsweep = np.zeros(shape=(n_samples,))
        sinsweep[0:sinsweep_hat.shape[0]] = sinsweep_hat

        # the convolutional inverse
        envelope = (w2/w1)**(-taxis)  # Holters2009, Eq.(9)
        invfilter = np.flipud(sinsweep)*envelope
        scaling = pi*n_samples * \
            (w1/w2-1)/(2*(w2-w1)*log(w1/w2)) * \
            (w2-w1)/pi  # Holters2009, Eq.10

        # fade-in window. Fade out removed because causes ringing - cropping at zero cross instead
        taperStart = signal.tukey(n_samples, 0)
        taperWindow = np.ones(shape=(n_samples,))
        taperWindow[0:int(n_samples/2)] = taperStart[0:int(n_samples/2)]
        sinsweep = sinsweep*taperWindow

        # Final excitation including repetition and pauses
        sinsweep = np.expand_dims(sinsweep, axis=1)
        zerostart = np.zeros(shape=(silence_at_start*fs, 1))
        zeroend = np.zeros(shape=(silence_at_end*fs, 1))
        sinsweep = np.concatenate(
            (np.concatenate((zerostart, sinsweep), axis=0), zeroend), axis=0)
        sinsweep = np.transpose(
            np.tile(np.transpose(sinsweep), n_repetitions))

        # Set the attributes
        self.total_duration = (silence_at_start + silence_at_end + n_seconds)*fs
        self.invfilter = invfilter/amplitude**2/scaling
        self.n_repetitions = n_repetitions
        self.signal = sinsweep
        self.times = np.arange(self.total_duration)/Fs
        self.n_seconds = n_seconds
        self.n_repetitions = n_repetitions
        self.time_ranges = [silence_at_start, silence_at_end]

        return self.times.copy(), self.signal.copy()


if if __name__ == "__main__":
    ps = ProbeSignal('exp_sine_sweep', 48000)
    t, s = ps.generate(10, 0.9, 3, 2, 2, [20, 20e3])

    import matplotlib.pyplot as plt
    plt.plot(t, s)
    plt.show()
    pass
