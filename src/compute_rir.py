import numpy as np
import scipy.signal as sg
import soundfile as sf
import pyroomacoustics as pra

import matplotlib.pyplot as plt

from stimulus import ProbeSignal
from risotto.deconvolution import matlab_deconvolution

# retrieve probe signal
Fs = 48000
ps = ProbeSignal('exp_sine_sweep', Fs)

n_seconds = 10
amplitude = 0.7
n_repetitions = 3
silence_at_start = 2
silence_at_end = 2
sweeprange = [100, 14e3]
times, signal = ps.generate(n_seconds, amplitude, n_repetitions,
                   silence_at_start, silence_at_end, sweeprange)

# rec, fs = sf.read('./data/recordings/2020-01-21__20-25-38_ch16.wav')
# rec = rec[:, 0]
# sf.write('./data/recordings/2020-01-21__20-25-38_ch16_mic1.wav', rec, fs)
rec, fs = sf.read('./data/recordings/2020-01-21__20-25-38_ch16_mic1.wav')
rec = rec[:, None]

mic_pos = [0.91171319, 3.91, 1.038]
src_pos = [3.65, 1.00, 1.415]
room_size = [5.741, 5.763, 2.353]

room = pra.ShoeBox(room_size, fs=48000, max_order=5)
room.add_microphone_array(pra.MicrophoneArray(np.array(mic_pos)[:, None], fs=room.fs))
room.add_source(src_pos)

room.image_source_model(use_libroom=False)

# room.plot_rir()
# plt.show()
K = 70
I = 1
J = 1
toa = np.zeros([I, J, K])
amp = np.zeros([I, J, K])
walls = np.zeros([I, J, K])
order = np.zeros([I, J, K])
for i in range(I):
    for j in range(1):
        images = room.sources[j].images
        center = room.mic_array.center
        distances = np.linalg.norm(
            images - room.mic_array.R[:, i, None], axis=0)
        # order in loc
        ordering = np.argsort(distances)[:K]
        for o, k in enumerate(ordering):
            amp[i, j, o] = room.sources[j].damping[k] / \
                (4 * np.pi * distances[k])
            toa[i, j, o] = distances[k]/340
            walls[i, j, o] = room.sources[j].walls[k]
            order[i, j, o] = room.sources[j].orders[k]

h = ps.compute_rir(rec)

np.save('./data/interim/rir.npy', h)

h_hrt = np.abs(sg.hilbert(h))
h_sqr = h**2

amp = amp / amp[0, 0, 0]
toa = Fs*(toa+0.13556)

# plt.plot(h_hrt/np.max(h_hrt))
plt.plot((h_sqr/np.max(h_sqr))[:20000], 'C1')

stem = plt.stem(toa[0, 0, :], amp[0, 0, :], linefmt='--', use_line_collection=True)
stem.stemlines.set_alpha(0.5)

for k in range(K):
    wall_idx = int(walls[0, 0, k])
    if wall_idx == -1:
        wall_name = 'direct'
    else:
        wall_name = '%s:%d' % (room.wall_names[wall_idx], order[0, 0, k])
    plt.text(toa[0, 0, k], amp[0, 0, k], wall_name)
plt.show()

