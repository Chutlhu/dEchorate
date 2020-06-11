#
# This class is used to perform the segmentation of early reflections given RIRs.
#
# Author: Luca Remaggi
# Email: l.remaggi@surrey.ac.uk
# 05/02/2018

import numpy as np

from scipy import signal
import numpy as np
from audiolazy import lazy_lpc as lpc


class Utility:

    def __init__(self, RIR=None, fs=None, x=None, m='b'):
        self.RIR = RIR
        self.fs = fs
        self.x = x
        self.m = m
        self.y = None
        self.toff = None
        self.tew = None
        self.sew = None
        self.RT = None
        self.EDC_log = None
        self.EDC = None

    def xewgrdel(self):
        # This is the DYPSA algorithm that has been translated from Matlab to Python. The DYPSA algorithm was first
        # presented in P. A. Naylor, A. Kounoudes, J. Gudnason and M. Brookes, ''Estimation of glottal closure instants in
        # voiced speech using the DYPSA algorithm'', IEEE Trans. on Audio, Speech and Lang. Proc., Vol. 15, No. 1, Jan. 2007

        # General variables
        dy_gwlen = 0.003
        dy_fwlen = 0.00045

        # Perform group delay calculation
        # Force window length to be odd
        gw = np.int_(2 * np.floor(dy_gwlen*self.fs/2) + 1)
        ghw = signal.hamming(gw, 1)
        ghwn = np.squeeze(ghw * [np.array(range(gw-1, -gw, -2))/2])

        RIR2 = self.RIR ** 2
        yn = signal.lfilter(ghwn, [1], RIR2)
        yd = signal.lfilter(ghw, [1], RIR2)
        yd[abs(yd) < 10**-16] = 10**-16  # It prevents infinity
        self.y = yn[gw-1:] / yd[gw-1:]
        self.toff = (gw - 1) / 2
        # Force window length to be odd
        fw = np.int_(2 * np.floor(dy_fwlen*self.fs/2) + 1)
        if fw > 1:
            daw = signal.hamming(fw, 1)
            self.y = signal.lfilter(
                daw, [1], self.y) / np.sum(daw)  # Low pass filtering
            self.toff = self.toff - (fw - 1)/2

        # Finding zero crossing
        self.x = self.y * 1
        self.m = 'n'
        self.zerocross()
        self.tew = self.t
        self.sew = self.s

        self.tew = self.tew + self.toff

        return self

    def zerocross(self):
        # This code is translated from the original version that was in Matlab, within the VOICEBOX toolbox
        # zerocross finds the zeros crossing in a signal

        self.x[abs(self.x) < 10 ** -5] = 0

        self.s = self.x >= 0
        # Converting false to 0 and true to 1
        self.s = np.int_(np.array(self.s))
        k = self.s[1:] - self.s[:-1]
        if self.m is 'p':
            f = np.where(k > 0)
        elif self.m is 'n':
            f = np.where(k < 0)
        else:
            f = k != 0

        f = np.transpose(np.int_(f))

        self.s = np.subtract(self.x[f + 1], self.x[f])
        self.t = f - np.divide(self.x[f], self.s)

        return self


class Peakpicking:

    def __init__(self, RIR, fs, groupdelay_threshold, use_LPC=1, cutoff_samples=5000, nLPC=12):
        self.RIR = RIR
        self.fs = fs
        self.groupdelay_threshold = groupdelay_threshold
        self.use_LPC = use_LPC
        self.cutoff_samples = cutoff_samples
        self.nLPC = nLPC
        self.p_pos = None

    def DYPSA(self):
        # This method estimates the position of peaks in a room impulse response by applying the DYPSA algorithm

        # Check that cutoff_samples is integer
        cutoff_samples = np.int_(self.cutoff_samples)

        # General variables internal to this method
        prev_rir = self.RIR * 1  # This is defined to allow future changes at the peak positions
        l_rir = len(self.RIR)
        internal_RIR = self.RIR * 1
        internal_RIR[cutoff_samples:l_rir] = 0

        if self.use_LPC == 1:
            # LPC for reduction of amount of data in RIR
            rir_up = signal.decimate(internal_RIR, 2)
            l_rir_lpc = len(rir_up)

            # Calculate the matching AR filter based on the RIR
            ar = lpc.lpc.kautocor(rir_up, self.nLPC)
            a = np.array(ar.numerator)
            b = np.array(ar.denominator)

            # Convert the filter into a time-reversed impulse response
            impulse = np.zeros(l_rir_lpc)
            impulse[0] = 1
            matched_forward = signal.lfilter(b, a, impulse)
            matched = np.flipud(matched_forward)

            # Apply the matched filter to the RIR
            rir_matched = signal.convolve(rir_up, matched)
            rir_matched = rir_matched[l_rir_lpc-1:]

            # Linearly interpolating
            RIR_new = signal.upfirdn([1], rir_matched, 2)

        # Realigning the new RIR with the original one
        val_max_new = np.argmax(abs(RIR_new))
        val_max_old = np.argmax(abs(prev_rir))
        diff_max = val_max_new - val_max_old
        if diff_max > 0:
            del internal_RIR
            internal_RIR = np.concatenate(
                [RIR_new[diff_max:], np.zeros(diff_max)])
        elif diff_max < 0:
            del internal_RIR
            internal_RIR = np.concatenate(
                [np.zeros(abs(diff_max)), RIR_new[:l_rir-abs(diff_max)]])
        else:
            del internal_RIR
            internal_RIR = RIR_new

        # Running the DYPSA algorithm
        OriginalDYPSA = Utility(RIR=internal_RIR, fs=self.fs)
        peaks_properties = OriginalDYPSA.xewgrdel()
        tew = peaks_properties.tew
        sew = peaks_properties.sew
        y = peaks_properties.y
        ntew = np.int_(np.round_(tew))

        # This avoids possible problems with sources too close to the microphones
        if ntew[0] < 0:
            ntew[0] = 0

        # Create an array which has zeros except for where tew defines peak positions
        k = 0
        peaks_init = np.zeros(l_rir)
        for idx_samp in range(0, len(y)):
            if k == len(ntew):
                break

            if idx_samp == ntew[k]:
                peaks_init[idx_samp] = 1
                k += 1

        # Peaks taken from the group-delay function where the slope is less than the threshold in input are deleted
        for idx_sew in range(0, len(sew)):
            if sew[idx_sew] > self.groupdelay_threshold:
                peaks_init[ntew[idx_sew]] = 0
        self.p_pos = peaks_init

        # Normalizing the RIR
        internal_RIR = abs(internal_RIR)
        norm_val = np.max(internal_RIR)
        internal_RIR = internal_RIR / norm_val

        # Take the neighborhood of the calculated position in the signal (which corresponds in total to 1ms) taking the rms
        # of the energy
        half_win = int(round(self.fs/2000))
        for idx_samp in range(0, len(ntew)):
            center = int(ntew[idx_samp])
            if (center - half_win) > 0:
                segment = internal_RIR[center-half_win:center+half_win]
            else:
                segment = internal_RIR[0:center+half_win]

            self.p_pos[center] = np.sqrt(np.mean(segment**2))

        ################################################################
        # From here there are additional improvements to the performance
        ################################################################
        # First, the array containing the peaks is normalized, and the position of the strongest peak found
        self.p_pos = self.p_pos / np.max(self.p_pos)
        ds_pos = int(np.argmax(self.p_pos))

        # Everything before the direct sound is equal to zero
        self.p_pos[:ds_pos-1] = 0

        # Deletes small errors by aligning the estimated direct sound position to the one in input
        ds_pos_gt = int(np.argmax(internal_RIR))
        estimation_err = ds_pos_gt - ds_pos
        if estimation_err > 0:
            self.p_pos = list(self.p_pos)
            self.p_pos = [[0]*estimation_err + self.p_pos[estimation_err:]]
        elif estimation_err < 0:
            self.p_pos = list(self.p_pos)
            self.p_pos = [
                self.p_pos[abs(estimation_err):] + [0]*abs(estimation_err)]
        self.p_pos = np.transpose(np.array(self.p_pos))

        return self


class Segmentation:

    def __init__(self, RIRs, fs, groupdelay_threshold, use_LPC, discrete_mode, nPeaks, hamm_lengths):
        self.RIRs = RIRs
        self.fs = fs
        self.groupdelay_threshold = groupdelay_threshold
        self.use_LPC = use_LPC
        self.discrete_mode = discrete_mode
        self.nPeaks = nPeaks
        self.segments = None
        self.TOAs_sample_single_mic = None
        self.hamm_lengths = hamm_lengths

    def segmentation(self):
        # Run DYPSA with the B-format omni component only (W channel)
        peakpicking = Peakpicking(RIR=self.RIRs[:, 0], fs=self.fs,
                                  groupdelay_threshold=self.groupdelay_threshold,
                                  use_LPC=self.use_LPC)
        peakpicking.DYPSA()
        p_pos = peakpicking.p_pos

        # Choosing which peaks to prioritize
        if self.discrete_mode is 'first':
            # Find peaks in the DYPSA output
            locs_all = np.transpose(np.array(np.where(p_pos[:, 0] != 0)))
            locs = locs_all[:(self.nPeaks + 5)]
            peaks = np.squeeze(p_pos[locs])
            firstearlypeaks = []
            firstearlylocs = []
        elif self.discrete_mode is 'strongest':
            # Find the first two in time
            locs_all = np.transpose(np.array(np.where(p_pos[:, 0] != 0)))
            firstearlylocs = locs_all[:2]
            firstearlypeaks = np.squeeze(p_pos[firstearlylocs])

            # Then finds the first peaks in energy-descending order
            peaks = np.squeeze(p_pos[locs_all])
            peaks = list(peaks)
            peaks = np.array(sorted(peaks, reverse=True))
            locs_mixed, idx_locs = np.where(p_pos == peaks)
            locs = locs_mixed[idx_locs]

        # Select the reflections TOAs
        first_and_strong = list(locs) + list(firstearlylocs)
        uniquelocs = np.unique(first_and_strong)
        self.TOAs_sample_single_mic = uniquelocs[0:self.nPeaks]

        # Create a dictionary and store inside the reflection segments
        self.segments = {'Direct_sound': self.RIRs[self.TOAs_sample_single_mic[0]-self.hamm_lengths[0]:
                                                   self.TOAs_sample_single_mic[0] + self.hamm_lengths[0], :]}
        for idx_refl in range(1, self.nPeaks):
            self.segments['Reflection' + str(idx_refl)] = self.RIRs[self.TOAs_sample_single_mic[idx_refl] -
                                                                    self.hamm_lengths[idx_refl]:self.TOAs_sample_single_mic[idx_refl] +
                                                                    self.hamm_lengths[idx_refl], :]

        return self
