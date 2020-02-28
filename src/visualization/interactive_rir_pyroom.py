import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D

import pyroomacoustics as pra


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
            'north': 0.3,
            'south': 0.3,
            'east': 0.3,
            'west': 0.3,
            'floor': 0.3,
            'ceiling': 0.3,
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


rir = RIR()

# Figure
fig = plt.figure(figsize=(16,9))
ax1 = fig.add_subplot(121, xlim=(-0.1, 1), ylim=(-0.1, 1))
ax2 = fig.add_subplot(122, projection='3d')
plt.subplots_adjust(bottom=0.3)

# Slider
ax_mic_x = plt.axes([0.1, 0.20, 0.35, 0.02]) # x, y, w, h
ax_mic_y = plt.axes([0.1, 0.15, 0.35, 0.02]) # x, y, w, h
ax_mic_z = plt.axes([0.1, 0.10, 0.35, 0.02]) # x, y, w, h

ax_src_x = plt.axes([0.6, 0.20, 0.35, 0.02])  # x, y, w, h
ax_src_y = plt.axes([0.6, 0.15, 0.35, 0.02])  # x, y, w, h
ax_src_z = plt.axes([0.6, 0.10, 0.35, 0.02])  # x, y, w, h

ax_speed = plt.axes([0.1, 0.05, 0.35, 0.02])  # x, y, w, h

sl_mic_x = Slider(ax_mic_x, 'mic_x', 0, rir.room_size[0], valinit=0.5, valstep=0.05)
sl_mic_y = Slider(ax_mic_y, 'mic_y', 0, rir.room_size[1], valinit=0.5, valstep=0.05)
sl_mic_z = Slider(ax_mic_z, 'mic_z', 0, rir.room_size[2], valinit=0.5, valstep=0.05)

sl_src_x = Slider(ax_src_x, 'src_x', 0, rir.room_size[0], valinit=3.5, valstep=0.05)
sl_src_y = Slider(ax_src_y, 'src_y', 0, rir.room_size[1], valinit=3.5, valstep=0.05)
sl_src_z = Slider(ax_src_z, 'src_z', 0, rir.room_size[2], valinit=3.5, valstep=0.05)

sl_speed = Slider(ax_speed, 'velocity', 335, 350, valinit=343, valstep=0.5)

def update(val=None):
    ax1.clear()
    ax2.clear()

    rir.set_c(sl_speed.val)
    rir.set_mic(sl_mic_x.val, sl_mic_y.val, sl_mic_z.val)
    rir.set_src(sl_src_x.val, sl_src_y.val, sl_src_z.val)

    amp, tau, _, _ = rir.get_note()
    times, h = rir.get_rir()

    rir.plot_room(ax2)
    ax1.plot(times, h, color='C1')
    ax1.stem(tau, amp, use_line_collection=True)
    ax1.set_xlim([-0.001, 0.03])
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

plt.show()
