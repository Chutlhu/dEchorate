import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from matplotlib.widgets import Slider, RadioButtons
from mpl_toolkits.mplot3d import Axes3D

from src.dataset import SyntheticDataset, DechorateDataset

import pyroomacoustics as pra

dataset_dir = './data/dECHORATE/'
path_to_processed = './data/processed/'
path_to_note_csv = dataset_dir + 'annotations/dECHORATE_database.csv'

imag_dset = SyntheticDataset()
real_dset = DechorateDataset(path_to_processed, path_to_note_csv)

imag_dset.set_room_size([5.741, 5.763, 2.353])
imag_dset.set_c(343)
imag_dset.set_k_order(3)
imag_dset.set_k_reflc(7)


real_dset.set_dataset('000000')
real_dset.set_entry(0, 0)
mic_pos, src_pos = real_dset.get_mic_and_src_pos()
times, h_rec = real_dset.get_rir()

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

sl_mic_x = Slider(ax_mic_x, 'mic_x', 0, imag_dset.room_size[0], valinit=mic_pos[0], valstep=0.05)
sl_mic_y = Slider(ax_mic_y, 'mic_y', 0, imag_dset.room_size[1], valinit=mic_pos[1], valstep=0.05)
sl_mic_z = Slider(ax_mic_z, 'mic_z', 0, imag_dset.room_size[2], valinit=mic_pos[2], valstep=0.05)

sl_src_x = Slider(ax_src_x, 'src_x', 0, imag_dset.room_size[0], valinit=src_pos[0], valstep=0.05)
sl_src_y = Slider(ax_src_y, 'src_y', 0, imag_dset.room_size[1], valinit=src_pos[1], valstep=0.05)
sl_src_z = Slider(ax_src_z, 'src_z', 0, imag_dset.room_size[2], valinit=src_pos[2], valstep=0.05)

sl_speed = Slider(ax_speed, 'velocity', 335, 350, valinit=343, valstep=0.5)
sl_sxlim = Slider(ax_sxlim, 'x ax start', -0.05, 0.20, valinit=-.001, valstep=0.005)
sl_fxlim = Slider(ax_fxlim, 'x ax end', -0.05, 0.20, valinit=0.03, valstep=0.005)


def update(val=None):
    ax1.clear()
    ax2.clear()

    real_dset.set_entry(0, 1)
    mic_pos, src_pos = real_dset.get_mic_and_src_pos()
    times, h_real = real_dset.get_rir()
    h_real = np.abs(h_real)/np.max(np.abs(h_real))
    ax1.plot(times, h_real, color='C0')
    ax2.scatter(mic_pos[0], mic_pos[1], mic_pos[2], marker='x', color='red')
    ax2.scatter(src_pos[0], src_pos[1], src_pos[2], marker='^', color='red')

    imag_dset.set_c(sl_speed.val)
    imag_dset.set_mic(sl_mic_x.val, sl_mic_y.val, sl_mic_z.val)
    imag_dset.set_src(sl_src_x.val, sl_src_y.val, sl_src_z.val)

    amp, tau, _, _ = imag_dset.get_note()
    times, h_imag = imag_dset.get_rir()

    imag_dset.plot_room(ax2)
    ax1.plot(times, h_imag, color='C1')
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
bt_dsets.on_clicked(lambda  val: update(real_dset.set_dataset(val)))

plt.show()
