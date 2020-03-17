import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pyroomacoustics as pra

from copy import deepcopy as cp

from matplotlib.widgets import Slider, RadioButtons
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
plt.subplots_adjust(bottom=0.15)
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

    print(curr_toa_note['toa'][0, i, j, 0])

    # plot the echo information
    for k in range(K):
        toa = curr_toa_note['toa'][k, i, j, 0]
        amp = curr_toa_note['amp'][k, i, j, 0]
        wall = curr_toa_note['wall'][k, i, j, 0]
        order = curr_toa_note['order'][k, i, j, 0]
        ax.axvline(x=int(toa*Fs), alpha=0.5)
        ax.text(toa*Fs, 0.025, r'$\tau_{%s}^{%d}$' %
                    (wall.decode(), order), fontsize=12)
    ax.set_xlim([0, 2000])
    ax.set_ylim([-0.05, 2.2])
    ax.set_title('RIRs dataset %s\nmic %d, src %d' % (datasets[d], i, j))


# Buttons
ax_which_tau = plt.axes([0.01, 0.15, 0.12, 0.40])  # x, y, w, h
bt_select_tau = RadioButtons(ax_which_tau, taus_list)

# Slider
ax_move_tau = plt.axes([0.1, 0.05, 0.80, 0.02])  # x, y, w, h
sl_move_tau = Slider(ax_move_tau, r'current $\tau$', 0, 2000, valinit=200, valstep=0.01)

status = {
    'idx' : 0,
    'val' : 0,
    'i' : i,
    'j' : j,
    'toa_note': cp(toa_note)
}

def update_taus(idx, val, toa_note, i, j):
    toa_note['toa'][idx, i, j, 0] = val
    return toa_note

def update(status, val=None, idx=None):
    ax1.clear()
    ax2.clear()

    if idx is not None:
        status['idx'] = int(idx.split(' ')[0])

    if val is not None:
        status['val'] = val/48000

    toa_note = update_taus(status['idx'], status['val'],
                           status['toa_note'], status['i'], status['j'])

    plot_rirs_and_note(all_rirs, toa_note, i, j, ax1)
    plot_rirs_and_note(all_rirs_clean, toa_note, i, j, ax2)

    fig.canvas.draw_idle()


# init
update(status)
sl_move_tau.on_changed(lambda val: update(status, val=val, idx=None))
bt_select_tau.on_clicked(lambda idx: update(status, val=None, idx=idx))

plt.show()
