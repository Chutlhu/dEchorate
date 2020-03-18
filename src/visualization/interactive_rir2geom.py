import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pyroomacoustics as pra

from copy import deepcopy as cp

from matplotlib.widgets import Slider, RadioButtons, Button
from mpl_toolkits.mplot3d import Axes3D

from src.dataset import SyntheticDataset, DechorateDataset
from src.utils.file_utils import load_from_pickle
from src.utils.dsp_utils import normalize


dataset_dir = './data/dECHORATE/'
path_to_processed = './data/processed/'
path_to_note_csv = dataset_dir + 'annotations/dECHORATE_database.csv'

imag_dset = SyntheticDataset()
real_dset = DechorateDataset(path_to_processed, path_to_note_csv)

params = {
    'Fs' : 48000,
}

i, j = (0, 0)

imag_dset.set_room_size([5.741, 5.763, 2.353])
imag_dset.set_c(343)
imag_dset.set_k_order(4)
imag_dset.set_k_reflc(15)
datasets = ['000000', '010000', '001000', '000100', '000010',
            '000001', '011000', '011100', '011110', '011111']

real_dset.set_dataset('000000')
real_dset.set_entry(i, j)
mic_pos, src_pos = real_dset.get_mic_and_src_pos()
times, h_rec = real_dset.get_rir()

print('mic_pos init', mic_pos)
print('src_pos init', src_pos)

# Figure
scaling = 1.1
fig = plt.figure(figsize=(16*scaling, 9*scaling))
ax1 = fig.add_subplot(121, xlim=(-0.1, 2), ylim=(-0.1, 1))
ax2 = fig.add_subplot(122, xlim=(-0.1, 2), ylim=(-0.1, 1), sharex=ax1)
# ax2 = fig.add_subplot(122, projection='3d')
plt.subplots_adjust(bottom=0.20)
plt.subplots_adjust(left=0.15)

all_rirs = np.load('./data/tmp/all_rirs.npy')
all_rirs_clean = np.load('./data/tmp/all_rirs_clean.npy')
toa_note = load_from_pickle('./data/tmp/toa_note.pkl')

L, I, J, D = all_rirs.shape
K, I, J, D = toa_note['toa'].shape
Fs = params['Fs']

taus_list = [r'%d $\tau$' % i for i in range(K)]


def plot_rirs_and_note(rirs, curr_toa_note, i, j, ax):
    for d in range(D):

        rir = rirs[:, i, j, d]

        # if d == 0:
        #     plt.plot(rir**2 + 0.2*d, alpha=.2, color='C1')
        rir_to_plot = (normalize(rir))**2
        rir_to_plot = np.clip(rir_to_plot, 0, 0.34)
        rir_to_plot = normalize(rir_to_plot)
        ax.plot(rir_to_plot + 0.2*d)

        # Print the dataset name
        wall_code_name = 'fcwsen'
        wall_code = [int(i) for i in list(datasets[d])]
        curr_walls = [wall_code_name[w]
                        for w, code in enumerate(wall_code) if code == 1]
        ax.text(50, 0.07 + 0.2*d, datasets[d])
        ax.text(50, 0.03 + 0.2*d, curr_walls)

    # plot the echo information
    for k in range(K):
        toa = curr_toa_note['toa'][k, i, j, 0]
        amp = curr_toa_note['amp'][k, i, j, 0]
        wall = curr_toa_note['wall'][k, i, j, 0]
        order = curr_toa_note['order'][k, i, j, 0]
        ax.axvline(x=int(toa*Fs), alpha=0.5)
        ax.text(toa*Fs, 0.025, r'$\tau_{%s}^{%d}$' %
                    (wall.decode(), order), fontsize=12)
    ax.set_ylim([-0.05, 2.2])
    ax.set_title('RIRs dataset %s\nmic %d, src %d' % (datasets[d], i, j))


# Buttons
ax_which_tau = plt.axes([0.01, 0.15, 0.12, 0.40])  # x, y, w, h
bt_select_tau = RadioButtons(ax_which_tau, taus_list)

ax_xlim0_prev = plt.axes([0.87, 0.04, 0.02, 0.02])
ax_xlim0_next = plt.axes([0.90, 0.04, 0.02, 0.02])
ax_xlim1_prev = plt.axes([0.87, 0.07, 0.02, 0.02])
ax_xlim1_next = plt.axes([0.90, 0.07, 0.02, 0.02])
ax_move_tau_prev = plt.axes([0.87, 0.01, 0.02, 0.02])
ax_move_tau_next = plt.axes([0.90, 0.01, 0.02, 0.02])
bt_xlim0_prev = Button(ax_xlim0_prev, '<')
bt_xlim0_next = Button(ax_xlim0_next, '>')
bt_xlim1_prev = Button(ax_xlim1_prev, '<')
bt_xlim1_next = Button(ax_xlim1_next, '>')
bt_move_tau_prev = Button(ax_move_tau_prev, '<')
bt_move_tau_next = Button(ax_move_tau_next, '>')

# Slider
ax_move_tau = plt.axes([0.1, 0.01, 0.70, 0.02])  # x, y, w, h
sl_move_tau = Slider(ax_move_tau, r'current $\tau$', 0, 2000, valinit=200, valstep=0.01)
ax_set_xlim0 = plt.axes([0.1, 0.04, 0.70, 0.02])  # x, y, w, h
ax_set_xlim1 = plt.axes([0.1, 0.07, 0.70, 0.02])  # x, y, w, h
sl_set_xlim0 = Slider(ax_set_xlim0, 'x min', 0, 7990, valinit=0, valstep=1)
sl_set_xlim1 = Slider(ax_set_xlim1, 'x max', 10, 8000, valinit=3000, valstep=1)



class Callbacks():

    def __init__(self, toa_note, i, j, fig, ax1, ax2):
        self.xlim = [0, 3000]
        self.toa_note = toa_note
        self.i = i
        self.j = j
        self.Fs = 48000

        self.ax1 = ax1
        self.ax2 = ax2
        self.fig = fig

        self.idx = 0
        self.val = self.toa_note['toa'][self.idx, self.i, self.j, 0]
        self._update()

    def set_idx(self, idx):
        if idx is not None:
            self.idx = int(idx.split(' ')[0])
        self._update_tau()
        self._update()

    def set_val(self, val=None, proc=None):
        if val is not None:
            self.val = val/self.Fs

        if proc is not None:
            self.val += proc/self.Fs

        self._update_tau()
        self._update()

    def set_xlim(self, i, val=None, proc=None):
        if val is not None:
            self.xlim[i] = val
        if proc is not None:
            self.xlim[i] += proc
        self._update()

    def _update_tau(self):
        self.toa_note['toa'][self.idx, self.i, self.j, 0] = self.val
        self._update()

    def _update(self):
        self.ax1.clear()
        self.ax2.clear()

        self.ax1.set_xlim(self.xlim)

        plot_rirs_and_note(all_rirs, self.toa_note, i, j, self.ax1)
        plot_rirs_and_note(all_rirs_clean, self.toa_note, i, j, self.ax2)

        self.fig.canvas.draw_idle()


# init
cb = Callbacks(cp(toa_note), i, j, fig, ax1, ax2)
sl_move_tau.on_changed(lambda val: cb.set_val(val))
sl_set_xlim0.on_changed(lambda val: cb.set_xlim(0, val))
sl_set_xlim1.on_changed(lambda val: cb.set_xlim(1, val))

bt_select_tau.on_clicked(lambda idx: cb.set_idx(idx=idx))
bt_move_tau_prev.on_clicked(lambda idx: cb.set_val(proc=-1))
bt_move_tau_next.on_clicked(lambda idx: cb.set_val(proc=+1))

bt_xlim0_prev.on_clicked(lambda x: cb.set_xlim(0, proc=-1))
bt_xlim0_next.on_clicked(lambda x: cb.set_xlim(0, proc=+1))
bt_xlim1_prev.on_clicked(lambda x: cb.set_xlim(1, proc=-1))
bt_xlim1_next.on_clicked(lambda x: cb.set_xlim(1, proc=+1))
plt.show()
