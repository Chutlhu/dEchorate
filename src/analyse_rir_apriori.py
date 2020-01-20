import numpy as np
import scipy as sp
import pyroomacoustics as pra
import matplotlib.pyplot as plt
import soundfile as sf

from src.utils.file_utils import load_from_mat

Fs = 48000
room_size = [5.741, 5.763, 2.353]

wall_absorption = {
    'north': 0.8,
    'south': 0.8,
    'east': 0.8,
    'west': 0.8,
    'ceiling': 0.2,
    'floor': 0.5,
}

room = pra.ShoeBox(room_size, fs=Fs, absorption=wall_absorption, max_order=7)
mics_filename = './data/raw/mics_position.csv'
mics = np.genfromtxt(mics_filename, delimiter=",")[:, 1:].T
mics = mics[:, 25:]
room.add_microphone_array(pra.MicrophoneArray(mics, room.fs))
print(mics)
_, I = mics.shape

srcs_filename = './data/raw/srcs_position.csv'
srcs = np.genfromtxt(srcs_filename, delimiter=",")[:, 1:].T
room.add_source(list(srcs[:, 1]))
print(srcs)
J = len(room.sources)

room.plot()
plt.show()

room.image_source_model(use_libroom=False)

K = 25 #len(room.sources[0].damping)
toa = np.zeros([I, J, K])
amp = np.zeros([I, J, K])
walls = np.zeros([I, J, K])
order = np.zeros([I, J, K])
for i in range(I):
    for j in range(J):
        images = room.sources[j].images
        center = room.mic_array.center
        distances = np.linalg.norm(images - room.mic_array.R[:, i, None], axis=0)
        # order in loc
        ordering = np.argsort(distances)[:K]
        for o, k in enumerate(ordering):
            amp[i, j, o] = room.sources[j].damping[k] / (4 * np.pi * distances[k])
            toa[i, j, o] = distances[k]/343
            walls[i, j, o] = room.sources[j].walls[k]
            order[i, j, o] = room.sources[j].orders[k]



h = load_from_mat('./data/recordings/rir_est_mic6_src_2.mat')['H']
times = np.arange(h.shape[0])/Fs
h = h[:, 0] / np.max(np.abs(h[:, 0]))
h *= amp[0, 0, 0]



plt.plot(times[6720:8720], np.abs(sp.signal.hilbert(h[6720:8720])))
plt.stem(toa[0, 0, :]+0.1345, amp[0, 0, :])
for k in range(K):
    wall_idx = int(walls[0, 0, k])
    if wall_idx == -1:
        wall_name = 'direct'
    else:
        wall_name = '%s:%d' % (room.wall_names[wall_idx], order[0, 0, k])
    plt.text(toa[0, 0, k]+0.1345, amp[0, 0, k], wall_name)
    print(toa[0, 0, k]+0.1345, amp[0, 0, k], wall_name)
plt.show()
