import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from matplotlib.widgets import Slider, RadioButtons
from mpl_toolkits.mplot3d import Axes3D

import pyroomacoustics as pra

dataset_dir = './data/dECHORATE/'
path_to_processed = './data/processed/'

path_to_note_csv = dataset_dir + 'annotations/dECHORATE_database.csv'

class RIR:
    def __init__(self):
        self.room_size = [6, 5, 4]
        self.x = [2, 2, 2]
        self.s = [4, 3, 1]
        self.Fs = 48000
        self.c = 343

        self.amp = None
        self.toa = None
        self.order = None

    def set_c(self, c):
        self.c = c

    def set_mic(self, x, y, z):
        self.x[0] = x
        self.x[1] = y
        self.x[2] = z

    def set_src(self, x, y, z):
        self.s[0] = x
        self.s[1] = y
        self.s[2] = z

    def make_room(self):
        absorption = {
            'north': 0.5,
            'south': 0.5,
            'east': 0.5,
            'west': 0.5,
            'floor': 0.5,
            'ceiling': 0.5,
        }

        room = pra.ShoeBox(
            self.room_size, fs=self.Fs,
            absorption=absorption, max_order=2)

        room.add_microphone_array(
            pra.MicrophoneArray(
                np.array(self.x)[:, None], room.fs))

        room.add_source(self.s)

        return room

    def plot_room(self, ax3D):
        ax3D.set_xlim([-0.1, self.room_size[0]+0.1])
        ax3D.set_ylim([-0.1, self.room_size[1]+0.1])
        ax3D.set_zlim([-0.1, self.room_size[2]+0.1])

        ax3D.scatter(self.x[0], self.x[1], self.x[2], marker='o', label='mics')
        ax3D.scatter(self.s[0], self.s[1], self.s[2], marker='x', label='srcs')

        # plot lines of the room
        ax3D.plot([0, self.room_size[0]], [0, 0], zs = [0, 0], color='black', alpha=0.6)
        ax3D.plot([0, 0], [0, self.room_size[1]], zs = [0, 0], color='black', alpha=0.6)
        ax3D.plot([0, 0], [0, 0], zs = [0, self.room_size[2]], color='black', alpha=0.6)

        # plot line of the image source
        # direct path
        ax3D.plot([self.x[0], self.s[0]], [self.x[1], self.s[1]], zs = [self.x[2], self.s[2]], c='C1', ls='--', alpha=0.8)

        return

    def get_rir(self):
        room = self.make_room()
        room.image_source_model()
        room.compute_rir()
        rir = room.rir[0][0]
        rir = rir[40:]
        return np.arange(len(rir))/self.Fs, rir/np.max(np.abs(rir))

    def get_note(self):
        room = self.make_room()
        room.image_source_model(use_libroom=False)

        j = 0
        K = 7
        toa = np.zeros(K)
        amp = np.zeros(K)
        walls = np.zeros(K)
        order = np.zeros(K)
        images = room.sources[j].images
        center = room.mic_array.center
        distances = np.linalg.norm(
            images - room.mic_array.R, axis=0)
        # order in loc
        ordering = np.argsort(distances)[:K]
        for o, k in enumerate(ordering):
            amp[o] = room.sources[j].damping[k] / (4 * np.pi * distances[k])
            toa[o] = distances[k]/self.c
            walls[o] = room.sources[j].walls[k]
            order[o] = room.sources[j].orders[k]

        amp = amp/amp[0]

        return amp, toa, walls, order


class REC():

    def __init__(self,):
        self.Fs = 48000
        self.dataset_data = None
        self.dataset_note = None
        self.entry = None
        self.mic_pos = None
        self.src_pos = None
        self.mic_i = 0
        self.src_j = 0
        self.rir = None

    def set_dataset(self, id):
        session_id = id
        path_to_data_hdf5 = path_to_processed + '%s_rir_data.hdf5' % session_id
        dset_rir = h5py.File(path_to_data_hdf5, 'r')
        dset_note = pd.read_csv(path_to_note_csv)
        f, c, w, s, e, n = [int(i) for i in list(session_id)]
        dset_note = dset_note.loc[
            (dset_note['room_rfl_floor'] == f)
            & (dset_note['room_rfl_ceiling'] == c)
            & (dset_note['room_rfl_west'] == w)
            & (dset_note['room_rfl_east'] == e)
            & (dset_note['room_rfl_north'] == n)
            & (dset_note['room_rfl_south'] == s)
            & (dset_note['room_fornitures'] == False)
            & (dset_note['src_signal'] == 'chirp')
        ]
        self.dataset_data = dset_rir
        self.dataset_note = dset_note

    def set_entry(self, i, j):
        self.mic_i = i
        self.src_j = j
        self.entry = self.dataset_note.loc[(self.dataset_note['src_id'] == j+1) & (self.dataset_note['mic_id'] == i+1)]

    def get_rir(self):
        wavefile = self.entry['filename'].values[0]
        rir = self.dataset_data['rir/%s/%d' % (wavefile, self.mic_i)][()].squeeze()
        rir_abs = np.abs(rir[6444:])
        self.rir = rir_abs/np.max(rir_abs)
        return np.arange(len(self.rir))/self.Fs, self.rir

    def get_mic_and_src_pos(self):
        self.mic_pos = np.array([self.entry['mic_pos_x'].values,
                        self.entry['mic_pos_y'].values, self.entry['mic_pos_z'].values]).squeeze()

        self.src_pos = np.array([self.entry['src_pos_x'].values,
                        self.entry['src_pos_y'].values, self.entry['src_pos_z'].values]).squeeze()
        return self.mic_pos, self.src_pos


rir = RIR()
rec = REC()

rec.set_dataset('000000')
rec.set_entry(0, 0)
mic_pos, src_pos = rec.get_mic_and_src_pos()
times, h_rec = rec.get_rir()

print('mic_pos init', mic_pos)
print('src_pos init', src_pos)

# Figure
fig = plt.figure(figsize=(16,9))
ax1 = fig.add_subplot(121, xlim=(-0.1, 2), ylim=(-0.1, 1))
ax2 = fig.add_subplot(122, projection='3d')
plt.subplots_adjust(bottom=0.3)
plt.subplots_adjust(left=0.2)

# Buttons
datasets = ('000000', '010000', '001000', '000100', '000010', '000001',
            '011000', '011100', '011110', '011111')
ax_radio = plt.axes([0.05, 0.6, 0.12, 0.34])
bt_dsets = RadioButtons(ax_radio, datasets)

# Slider
ax_mic_x = plt.axes([0.1, 0.20, 0.35, 0.02]) # x, y, w, h
ax_mic_y = plt.axes([0.1, 0.15, 0.35, 0.02]) # x, y, w, h
ax_mic_z = plt.axes([0.1, 0.10, 0.35, 0.02]) # x, y, w, h

ax_src_x = plt.axes([0.6, 0.20, 0.35, 0.02])  # x, y, w, h
ax_src_y = plt.axes([0.6, 0.15, 0.35, 0.02])  # x, y, w, h
ax_src_z = plt.axes([0.6, 0.10, 0.35, 0.02])  # x, y, w, h

ax_speed = plt.axes([0.1, 0.05, 0.35, 0.02])  # x, y, w, h
ax_sxlim = plt.axes([0.6, 0.05, 0.10, 0.02])  # x, y, w, h
ax_fxlim = plt.axes([0.8, 0.05, 0.10, 0.02])  # x, y, w, h

sl_mic_x = Slider(ax_mic_x, 'mic_x', 0, rir.room_size[0], valinit=mic_pos[0], valstep=0.05)
sl_mic_y = Slider(ax_mic_y, 'mic_y', 0, rir.room_size[1], valinit=mic_pos[1], valstep=0.05)
sl_mic_z = Slider(ax_mic_z, 'mic_z', 0, rir.room_size[2], valinit=mic_pos[2], valstep=0.05)

sl_src_x = Slider(ax_src_x, 'src_x', 0, rir.room_size[0], valinit=src_pos[0], valstep=0.05)
sl_src_y = Slider(ax_src_y, 'src_y', 0, rir.room_size[1], valinit=src_pos[1], valstep=0.05)
sl_src_z = Slider(ax_src_z, 'src_z', 0, rir.room_size[2], valinit=src_pos[2], valstep=0.05)

sl_speed = Slider(ax_speed, 'velocity', 335, 350, valinit=343, valstep=0.5)
sl_sxlim = Slider(ax_sxlim, 'x ax start', -0.05, 0.20, valinit=-.001, valstep=0.005)
sl_fxlim = Slider(ax_fxlim, 'x ax end', -0.05, 0.20, valinit=0.03, valstep=0.005)


def update(val=None):
    ax1.clear()
    ax2.clear()

    rec.set_entry(0, 0)
    mic_pos, src_pos = rec.get_mic_and_src_pos()
    times, h_rec = rec.get_rir()
    ax1.plot(times, h_rec, color='C0')
    ax2.scatter(mic_pos[0], mic_pos[1], mic_pos[2], marker='x', color='red')
    ax2.scatter(src_pos[0], src_pos[1], src_pos[2], marker='^', color='red')

    rir.set_c(sl_speed.val)
    rir.set_mic(sl_mic_x.val, sl_mic_y.val, sl_mic_z.val)
    rir.set_src(sl_src_x.val, sl_src_y.val, sl_src_z.val)

    amp, tau, _, _ = rir.get_note()
    times, h = rir.get_rir()

    rir.plot_room(ax2)
    ax1.plot(times, h, color='C1')
    ax1.stem(tau, amp, use_line_collection=True)
    ax1.set_xlim([sl_sxlim.val, sl_fxlim.val])
    ax1.set_ylim([-0.01, 1.01])

    fig.canvas.draw_idle()

# init
update()
sl_mic_x.on_changed(update)
sl_mic_y.on_changed(update)
sl_mic_z.on_changed(update)
sl_src_x.on_changed(update)
sl_src_y.on_changed(update)
sl_src_z.on_changed(update)
sl_speed.on_changed(update)
sl_sxlim.on_changed(update)
sl_fxlim.on_changed(update)
bt_dsets.on_clicked(lambda  val: update(rec.set_dataset(val)))

plt.show()
