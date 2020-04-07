import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pyroomacoustics as pra

from copy import deepcopy as cp

from matplotlib.widgets import Slider, RadioButtons, Button, TextBox, CheckButtons
from mpl_toolkits.mplot3d import Axes3D

from src.dataset import SyntheticDataset, DechorateDataset
from src.utils.file_utils import load_from_pickle, save_to_pickle
from src.utils.dsp_utils import normalize


dataset_dir = './data/dECHORATE/'
path_to_processed = './data/processed/'
path_to_note_csv = dataset_dir + 'annotations/dECHORATE_database.csv'

params = {
    'Fs' : 48000,
}
datasets = ['000000', '010000', '011000', '011100', '011110', '011111',
            '001000', '000100', '000010', '000001']


## INITIALIZE FIGURE
scaling = 0.8
fig = plt.figure(figsize=(16*scaling, 9*scaling))
ax1 = fig.add_subplot(221, xlim=(-0.1, 2), ylim=(-0.1, 1))
ax2 = fig.add_subplot(222, xlim=(-0.1, 2), ylim=(-0.1, 1), sharex=ax1, sharey=ax1)
ax3 = fig.add_subplot(212, xlim=(-0.1, 2), ylim=(-0.1, 1), sharex=ax1, sharey=ax1)

plt.subplots_adjust(top=0.90)
plt.subplots_adjust(left=0.15)
plt.subplots_adjust(right=0.85)

all_rirs = np.load('./data/tmp/all_rirs.npy')
toa_note = load_from_pickle('./data/tmp/toa_note.pkl')

L, I, J, D = all_rirs.shape
K, I, J, D = toa_note['toa'].shape
Fs = params['Fs']

taus_list = [r'%d $\tau_{%s}^{%d}$' % (k, toa_note['wall'][k, 0, 0, 0].decode(), toa_note['order'][k, 0, 0, 0]) for k in range(K)]


def plot_rirs_and_note(rirs, note, i, j, selected_k, ax, visibility):
    for d in range(D):
        if visibility[d]:
            rir_to_plot = rirs[:, d]
            rir_to_plot = rir_to_plot / np.max(np.abs(rir_to_plot), axis=0)

            ax.plot(rir_to_plot**2, alpha=.8, label=datasets[d])

        # # Print the dataset name
        # wall_code_name = 'fcwsen'
        # wall_code = [int(i) for i in list(datasets[d])]
        # curr_walls = [wall_code_name[w]
        #                 for w, code in enumerate(wall_code) if code == 1]
        # ax.annotate(datasets[d], [50, 0.07])
        # ax.annotate('|'.join(curr_walls), [50, 0.025])

    # plot the echo information
    for k in range(K):
        toa = note['toa'][k, i, j, 0]
        amp = note['amp'][k, i, j, 0]
        wall = note['wall'][k, i, j, 0]
        order = note['order'][k, i, j, 0]
        if k == selected_k:
            ax.axvline(x=int(toa*Fs), c='C5', alpha=1)
        else:
            ax.axvline(x=int(toa*Fs), c='C0', alpha=0.5)

        ax.annotate( r'$\tau_{%s}^{%d}$' % (wall.decode(), order), [toa*Fs, 0.25], fontsize=12)

    ax.set_ylim([-0.05, 2.2])
    ax.set_title('RIRs dataset %s\nmic %d, src %d' % (datasets[0], i, j))
    pass


def plot_rirs_and_note_merged(rirs, note, i, j, selected_k, ax, fun= lambda x : np.mean(x**2, axis=-1)):

    mean_rirs_to_plot = fun(rirs)
    mean_rirs_to_plot = normalize(mean_rirs_to_plot)
    ax.plot(mean_rirs_to_plot, alpha=1, color='red')

    # plot the echo information
    for k in range(K):
        toa = note['toa'][k, i, j, 0]
        wall = note['wall'][k, i, j, 0]
        order = note['order'][k, i, j, 0]
        if k == selected_k:
            ax.axvline(x=int(toa*Fs), c='C5', alpha=1)
        else:
            ax.axvline(x=int(toa*Fs), c='C0', alpha=0.5)

        ax.annotate(r'$\tau_{%s}^{%d}$' % (wall.decode(), order), [
                    toa*Fs, 0.25], fontsize=12)

    ax.set_ylim([-0.05, 1.1])
    pass


## EDIT PLOT VISIBILITY
rax = plt.axes([0.01, 0.6, 0.1, 0.30])
check = CheckButtons(rax, datasets, [True for d in datasets])

## EDIT TAU LOCATIONS
txt_boxes_tau = []
for k in range(7):
    rax = plt.axes([0.93, 0.8 - k*0.035, 0.05, 0.03])
    tbox = TextBox(rax, r'Set $\tau_{%s}^{%d}$  ' % (toa_note['wall'][k, 0, 0, 0].decode(), toa_note['order'][k, 0, 0, 0]),
                   initial=str(np.round(Fs*toa_note['toa'][k, 0, 0, 0], 2)))
    txt_boxes_tau.append(tbox)

## MAKE DIRECT PATH DECONVOLUTION
rax = plt.axes([0.93, 0.95, 0.05, 0.03])
dp_box1 = TextBox(rax, 'Before', initial='10')
rax = plt.axes([0.93, 0.915, 0.05, 0.03])
dp_box2 = TextBox(rax, 'After', initial='20')
rax = plt.axes([0.93, 0.88, 0.05, 0.03])
dp_box2 = Button(rax, 'Do it')

# Buttons
# ax_save_tau = plt.axes([0.01, 0.90, 0.04, 0.02])  # x, y, w, h
# bt_save_tau = Button(ax_save_tau, 'save')

# ax_which_tau = plt.axes([0.01, 0.15, 0.12, 0.40])  # x, y, w, h
# bt_select_tau = RadioButtons(ax_which_tau, taus_list)

# ax_load_tau = plt.axes([0.05, 0.90, 0.04, 0.02])  # x, y, w, h
# bt_load_tau = Button(ax_load_tau, 'load')
# ax_reset_curr_tau = plt.axes([0.01, 0.87, 0.04, 0.02])  # x, y, w, h
# bt_reset_curr_tau = Button(ax_reset_curr_tau, r'reset $\tau$')

# ax_xlim0_pprev = plt.axes([0.87, 0.04, 0.02, 0.02])
# bt_xlim0_pprev = Button(ax_xlim0_pprev, '<<')
# ax_xlim0_prev = plt.axes([0.89, 0.04, 0.02, 0.02])
# bt_xlim0_prev = Button(ax_xlim0_prev, '<')
# ax_xlim0_next = plt.axes([0.915, 0.04, 0.02, 0.02])
# bt_xlim0_next = Button(ax_xlim0_next, '>')
# ax_xlim0_nnext = plt.axes([0.935, 0.04, 0.02, 0.02])
# bt_xlim0_nnext = Button(ax_xlim0_nnext, '>>')

# ax_xlim1_pprev = plt.axes([0.87, 0.07, 0.02, 0.02])
# ax_xlim1_prev = plt.axes([0.89, 0.07, 0.02, 0.02])
# ax_xlim1_next = plt.axes([0.915, 0.07, 0.02, 0.02])
# ax_xlim1_nnext = plt.axes([0.935, 0.07, 0.02, 0.02])
# bt_xlim1_pprev = Button(ax_xlim1_pprev, '<<')
# bt_xlim1_next = Button(ax_xlim1_next, '>')
# bt_xlim1_prev = Button(ax_xlim1_prev, '<')
# bt_xlim1_nnext = Button(ax_xlim1_nnext, '>>')

# ax_move_tau_pprev = plt.axes([0.87, 0.01, 0.02, 0.02])
# ax_move_tau_prev = plt.axes([0.89, 0.01, 0.02, 0.02])
# ax_move_tau_next = plt.axes([0.915, 0.01, 0.02, 0.02])
# ax_move_tau_nnext = plt.axes([0.935, 0.01, 0.02, 0.02])
# bt_move_tau_pprev = Button(ax_move_tau_pprev, '<<')
# bt_move_tau_prev = Button(ax_move_tau_prev, '<')
# bt_move_tau_next = Button(ax_move_tau_next, '>')
# bt_move_tau_nnext = Button(ax_move_tau_nnext, '>>')

# # Slider
# ax_move_tau = plt.axes([0.1, 0.01, 0.70, 0.02])  # x, y, w, h
# sl_move_tau = Slider(ax_move_tau, r'current $\tau$', 0, 8000, valinit=200, valstep=0.01)
# ax_set_xlim0 = plt.axes([0.1, 0.04, 0.70, 0.02])  # x, y, w, h
# ax_set_xlim1 = plt.axes([0.1, 0.07, 0.70, 0.02])  # x, y, w, h
# sl_set_xlim0 = Slider(ax_set_xlim0, 'x min', 0, 7990, valinit=0, valstep=1)
# sl_set_xlim1 = Slider(ax_set_xlim1, 'x max', 10, 8000, valinit=3000, valstep=1)

# # Text
# ax_text_tau = plt.axes([0.05, 0.84, 0.04, 0.02])  # x, y, w, h
# tx_text_tau = TextBox(ax_text_tau, 'Set tau')
# ax_text_mic = plt.axes([0.05, 0.81, 0.04, 0.02])  # x, y, w, h
# tx_text_mic = TextBox(ax_text_mic, 'Set mic')
# ax_text_src = plt.axes([0.05, 0.78, 0.04, 0.02])  # x, y, w, h
# tx_text_src = TextBox(ax_text_src, 'Set src')


class Callbacks():

    def __init__(self, toa_note, rirs, all_rirs_clean, datasets, fig, axes=[ax1, ax2]):
        self.xlim = [0, 3000]
        self.all_rirs = rirs
        self.all_rirs_clean = all_rirs_clean
        self.curr_rirs = rirs[:, 0, 0, :]
        self.curr_rirs_clean = all_rirs_clean[:, 0, 0, :]
        self.toa_note = toa_note
        self.toa_note_backup = cp(toa_note)
        self.i = 0
        self.j = 0
        self.Fs = 48000
        self.k = 0

        self.datasets = datasets
        self.visibility = [True for d in datasets]

        self.axes = axes
        self.fig = fig

        self.idx = 0
        self.val = self.toa_note['toa'][self.idx, self.i, self.j, 0]
        self._update()

    def save_tau_to_file(self, foo):
        path = './data/interim/manual_annotation/'
        filename = 'src-%d-mic-%d.pkl' % (self.j, self.i)
        save_to_pickle(path + filename, self.toa_note)
        print('Annotation saved to file:', filename)

    def load_tau_note(self, bar):
        path = './data/interim/manual_annotation/'
        filename = 'src-%d-mic-%d.pkl' % (self.j, self.i)
        self.toa_note = load_from_pickle(path + filename)
        print('Annotation loaded from:', filename)
        self._update()

    def set_mic(self, i):
        self.i = i
        self._update()

    def set_src(self, j):
        self.j = j
        self._update()

    def set_idx(self, idx):
        if idx is not None:
            self.idx = int(idx.split(' ')[0])

        tau = self.toa_note['toa'][self.idx, self.i, self.j, 0]*self.Fs
        self.xlim = [tau-100, tau+100]
        self._update()

    def set_val(self, val=None, proc=None):
        if val is not None:
            self.val = val/self.Fs

        if proc is not None:
            self.val += proc/self.Fs

        self._update()

    def set_xlim(self, i, val=None, proc=None):
        if val is not None:
            self.xlim[i] = val
        if proc is not None:
            self.xlim[i] += proc
        self._update()

    def set_visibility(self, label):
        index = self.datasets.index(label)
        self.visibility[index] = not self.visibility[index]
        self._update()

    def reset_tau(self, bar):
        self.toa_note['toa'][self.idx, self.i, self.j, 0] =  \
            self.toa_note_backup['toa'][self.idx, self.i, self.j, 0]
        self._update()

    def _update(self):
        # update taus
        self.toa_note['toa'][self.idx, self.i, self.j, :] = self.val
        # update current rirs
        self.curr_rirs = self.all_rirs[:, self.i, self.j, :]
        self.curr_rirs_clean = self.all_rirs_clean[:, self.i, self.j, :]
        print(self.i, self.j)

        ## UPDATE PLOTS
        # clear axes
        self.axes[0].clear()
        self.axes[1].clear()
        self.axes[2].clear()

        self.axes[0].set_xlim(self.xlim)

        plot_rirs_and_note(self.curr_rirs, self.toa_note, self.i, self.j, self.idx, self.axes[0], self.visibility)
        plot_rirs_and_note(self.curr_rirs_clean, self.toa_note, self.i, self.j, self.idx, self.axes[1], self.visibility)

        # # peak picking of 2nd axis
        # mean_rirs_to_plot = np.max(self.curr_rirs_clean**2, axis=-1)
        # mean_rirs_to_plot = normalize(mean_rirs_to_plot)
        # peaks, _ = sp.signal.find_peaks(mean_rirs_to_plot, prominence=0.05, distance=10)
        # print(peaks)
        # self.ax2.scatter(peaks, mean_rirs_to_plot[peaks], color='red', marker='x')

        plot_rirs_and_note_merged(self.curr_rirs_clean, self.toa_note, self.i, self.j, self.idx, self.axes[2])

        self.fig.canvas.draw_idle()


# init
cb = Callbacks(cp(toa_note), all_rirs, all_rirs, datasets, fig, axes=[ax1, ax2, ax3])

check.on_clicked(cb.set_visibility)

# sl_move_tau.on_changed(lambda val: cb.set_val(val))
# sl_set_xlim0.on_changed(lambda val: cb.set_xlim(0, val))
# sl_set_xlim1.on_changed(lambda val: cb.set_xlim(1, val))

# bt_select_tau.on_clicked(lambda idx: cb.set_idx(idx=idx))
# bt_move_tau_pprev.on_clicked(lambda idx: cb.set_val(proc=-1))
# bt_move_tau_prev.on_clicked(lambda idx: cb.set_val(proc=-.1))
# bt_move_tau_next.on_clicked(lambda idx: cb.set_val(proc=+.1))
# bt_move_tau_nnext.on_clicked(lambda idx: cb.set_val(proc=+1))

# bt_xlim0_pprev.on_clicked(lambda x: cb.set_xlim(0, proc=-5))
# bt_xlim0_prev.on_clicked(lambda x: cb.set_xlim(0, proc=-1))
# bt_xlim0_next.on_clicked(lambda x: cb.set_xlim(0, proc=+1))
# bt_xlim0_nnext.on_clicked(lambda x: cb.set_xlim(0, proc=+5))

# bt_xlim1_pprev.on_clicked(lambda x: cb.set_xlim(1, proc=-5))
# bt_xlim1_prev.on_clicked(lambda x: cb.set_xlim(1, proc=-1))
# bt_xlim1_next.on_clicked(lambda x: cb.set_xlim(1, proc=+1))
# bt_xlim1_nnext.on_clicked(lambda x: cb.set_xlim(1, proc=+5))

# bt_save_tau.on_clicked(cb.save_tau_to_file)
# bt_load_tau.on_clicked(cb.load_tau_note)
# bt_reset_curr_tau.on_clicked(cb.reset_tau)

# tx_text_mic.on_submit(lambda i: cb.set_mic(int(i)))
# tx_text_src.on_submit(lambda j: cb.set_src(int(j)))

cb.set_xlim(0, 20)
cb.set_xlim(1,1000)
plt.show()
