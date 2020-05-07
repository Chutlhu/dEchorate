import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pyroomacoustics as pra
import peakutils as pk

from copy import deepcopy as cp
from datetime import datetime

from matplotlib.widgets import Slider, RadioButtons, Button, TextBox, CheckButtons
from mpl_toolkits.mplot3d import Axes3D

from src import constants
from src.dataset import SyntheticDataset, DechorateDataset
from src.utils.file_utils import load_from_pickle, save_to_pickle
from src.utils.dsp_utils import normalize

from tkinter.filedialog import askopenfilename

dataset_dir = './data/dECHORATE/'
path_to_processed = './data/processed/'
path_to_note_csv = dataset_dir + 'annotations/dECHORATE_database.csv'

params = {
    'Fs' : constants['Fs'],
}
datasets = constants['datasets']
curr_reflectors = constants['refl_order_calibr'][:7]
refl_order = constants['refl_order_pyroom']
Fs = constants['Fs']
recording_offset = constants['recording_offset']

## INITIALIZE FIGURE
scaling = 0.8
fig = plt.figure(figsize=(16*scaling, 9*scaling))
ax1 = fig.add_subplot(111)
plt.subplots_adjust(left=0.15, bottom=0.05, right=0.95, top=0.95)


path_to_after_calibration = './data/processed/post2_calibration/calib_output_mics_srcs_pos.pkl'
note_dict = load_from_pickle(path_to_after_calibration)

rirs = note_dict['rirs']
mics = note_dict['mics']
srcs = note_dict['srcs']
toas = note_dict['toa_pck']

L, Ir, Jr = rirs.shape
K, It, Jt = toas.shape
Dm, Im = mics.shape
Ds, Js = srcs.shape

assert Ir == Im == It
assert Jr == Jt == Js
assert Dm == Ds

def plot_rir_skyline(ax, rirs, toa_pck=None, toa_sym=None):

    L, I, J = rirs.shape
    rirs_skyline = rirs.transpose([0, 2, 1]).reshape([L, I*J])
    ax.imshow(rirs_skyline, extent=[0, I*J, 0, L], aspect='auto')

    # plot srcs boundaries
    for j in range(J):
        ax.axvline(j*30, color='C7')
    # plot time of emission
    ax.axhline(y=L-recording_offset, label='Time of Emission')

    for k in range(7):
        print(curr_reflectors)
        wall = curr_reflectors[k]
        r = refl_order.index(wall)
        # plot peak annotation
        if toa_pck is not None:
            ax.scatter(np.arange(I*J)+0.5, L - recording_offset - toa_peak[r, :, :].T.flatten()*Fs, c='C%d' % (k+2), marker='x', label='%s Picking' % wall)
        # plot simulated peak
        if toa_sym is not None:
            ax.scatter(np.arange(I*J)+0.5, L - recording_offset - toa_sym[r, :, :].T.flatten()*Fs, marker='o', facecolors='none', edgecolors='C%d' % (k+2), label='%s Pyroom' % wall)

    ax.set_ylim([18500, L - recording_offset])
    ax.set_xlim([0, I*J])
    ax.legend()
    return

plot_rir_skyline(ax1, rirs)


class Callbacks():

    def __init__(self, ):
        pass



# ## EDIT PLOT VISIBILITY
# rax = plt.axes([0.01, 0.6, 0.1, 0.30])
# initial_validity = [True if d == 1 else False for d, dataset in enumerate(datasets)]
# check = CheckButtons(rax, datasets, initial_validity)

# ## EDIT TAU LOCATIONS
# txt_boxes_tau = []
# for k in range(7):
#     rax = plt.axes([0.93, 0.8 - k*0.035, 0.05, 0.03])
#     tbox = TextBox(rax, r'Set $\tau_{%s}^{%d}$  ' % (toa_note['wall'][k, 0, 0, 0], toa_note['order'][k, 0, 0, 0]),
#                    initial=str(np.round(Fs*toa_note['toa'][k, 0, 0, 0], 2)))
#     txt_boxes_tau.append(tbox)

# ## MAKE DIRECT PATH DECONVOLUTION
# rax = plt.axes([0.93, 0.95, 0.05, 0.03])
# dp_box1 = TextBox(rax, 'Before', initial='10')
# rax = plt.axes([0.93, 0.915, 0.05, 0.03])
# dp_box2 = TextBox(rax, 'After', initial='20')
# rax = plt.axes([0.93, 0.88, 0.05, 0.03])
# dp_box = Button(rax, 'Do it')

# # Buttons
# ax_save_tau = plt.axes([0.88, 0.05, 0.10, 0.02])  # x, y, w, h
# bt_save_tau = Button(ax_save_tau, 'save tau to csv')
# ax_load_tau = plt.axes([0.88, 0.08, 0.10, 0.02])  # x, y, w, h
# bt_load_tau = Button(ax_load_tau, 'load tau csv')

# ax_peak_tau = plt.axes([0.88, 0.45, 0.10, 0.02])  # x, y, w, h
# bt_peak_tau = Button(ax_peak_tau, 'Run peak')

# # Text
# ax_text_mic = plt.axes([0.06, 0.95, 0.05, 0.03])  # x, y, w, h
# tx_text_mic = TextBox(ax_text_mic, 'Set mic ', initial='0')
# ax_text_src = plt.axes([0.06, 0.915, 0.05, 0.03])  # x, y, w, h
# tx_text_src = TextBox(ax_text_src, 'Set src ', initial='0')

# rax = plt.axes([0.93, 0.200, 0.05, 0.03])  # x, y, w, h
# txt_boxes_pck_dst = TextBox(rax, 'distance ', initial='10')
# rax = plt.axes([0.93, 0.235, 0.05, 0.03])  # x, y, w, h
# txt_boxes_pck_thr = TextBox(rax, 'threshold', initial='0.03')
# rax = plt.axes([0.93, 0.270, 0.05, 0.03])  # x, y, w, h
# txt_boxes_pck_num = TextBox(rax, 'num_peaks', initial='10')

# class Callbacks():

#     def __init__(self, toa_note, rirs, all_rirs_clean, datasets, initial_validity, fig, axes=[ax1, ax2]):
#         self.xlim = [0, 3000]
#         self.all_rirs = rirs
#         self.all_rirs_clean = all_rirs_clean
#         self.curr_rirs = rirs[:, 0, 0, :]
#         self.curr_rirs_clean = all_rirs_clean[:, 0, 0, :]
#         self.toa_note = toa_note
#         self.toa_note_backup = cp(toa_note)
#         self.i = 0
#         self.j = 0
#         self.Fs = 48000
#         self.k = 0
#         self.interpolated_peaks = None

#         self.peak = {
#             'distance' : 10,
#             'threshold' : 0.03,
#             'num_peaks' : 10,
#         }

#         self.feature = None

#         self.dp_extreme = [int(self.toa_note['toa'][0, self.i, self.j, 0]*self.Fs)-10,
#                            int(self.toa_note['toa'][0, self.i, self.j, 0]*self.Fs)-30]

#         self.datasets = datasets
#         self.visibility = initial_validity

#         self.axes = axes
#         self.fig = fig

#         self.idx = 0
#         self.val = self.toa_note['toa'][self.idx, self.i, self.j, 0]
#         self._update()


#     def save_tau_to_file(self, event):
#         path = './data/interim/manual_annotation/'
#         filename = "%s_gui_annotation" % datetime.now().strftime('%Y%m%d_%Hh%M')
#         save_to_pickle(path + filename + '.pkl', self.toa_note)
#         foo = self.toa_note['toa'][:7,:,:,0].reshape([7,I*J])
#         np.savetxt(path+filename+'.csv', foo, delimiter=',')
#         print('Annotation saved to file:', path+filename)
#         pass


#     def load_tau_note(self, bar):
#         filename = askopenfilename()
#         print(self.toa_note['toa'][:, self.i, self.j, 0])
#         self.toa_note = load_from_pickle(filename)
#         print('Annotation loaded from:', filename)
#         print(self.toa_note['toa'][:, self.i, self.j, 0])
#         self._update()


#     def set_mic(self, i):
#         self.i = i
#         self._update()


#     def set_src(self, j):
#         self.j = j
#         self._update()


#     def set_idx(self, idx):
#         if idx is not None:
#             self.idx = int(idx.split(' ')[0])

#         tau = self.toa_note['toa'][self.idx, self.i, self.j, 0]*self.Fs
#         self.xlim = [tau-100, tau+100]
#         self._update()


#     def set_val(self, val=None, proc=None):
#         if val is not None:
#             self.val = val/self.Fs

#         if proc is not None:
#             self.val += proc/self.Fs

#         self._update()

#     def set_extreme_dp_deconvolution(self, x, i):
#         self.dp_extreme[i] = x
#         print('Set %d extreme to %d' % (i, x))
#         self._update()
#         self.axes[0].fill_between(np.arange(self.dp_extreme[0], self.dp_extreme[1]), -0.2, 1.2, color='C8', alpha=0.2)
#         self.axes[1].fill_between(np.arange(self.dp_extreme[0], self.dp_extreme[1]), -0.2, 1.2, color='C8', alpha=0.2)
#         self.axes[2].fill_between(np.arange(self.dp_extreme[0], self.dp_extreme[1]), -0.2, 1.2, color='C8', alpha=0.2)



#     def set_xlim(self, i, val=None, proc=None):
#         if val is not None:
#             self.xlim[i] = val
#         if proc is not None:
#             self.xlim[i] += proc
#         self._update()


#     def set_visibility(self, label):
#         index = self.datasets.index(label)
#         self.visibility[index] = not self.visibility[index]
#         self._update()


#     def reset_tau(self, bar):
#         self.toa_note['toa'][self.idx, self.i, self.j, 0] =  \
#             self.toa_note_backup['toa'][self.idx, self.i, self.j, 0]
#         self._update()

#     def set_toa(self):
#         # update taus
#         self.toa_note['toa'][self.idx, self.i, self.j, :] = self.val

#     def set_tau_val(self, val, i):
#         # update taus
#         self.idx = i
#         self.toa_note['toa'][self.idx, self.i, self.j, :] = val/self.Fs
#         tmp = self.toa_note['toa'][self.idx, self.i, self.j, 0]
#         print('Updated tau %d to %1.2f [s] %1.6f' %
#               (self.idx, tmp, tmp*self.Fs))
#         self._update()


#     def set_peak_dict(self, key, val):
#         self.peak[key] = val
#         print('Registered change', key, self.peak[key])


#     def peak_peaking(self, event):
#         # run peak piking of the clean rirs
#         print('Run peak picking')
#         print(self.feature)
#         if self.feature is None:
#             pass
#         cb = cp(self.feature).squeeze()
#         peaks = pk.indexes(cb, thres=self.peak['threshold'], min_dist=self.peak['distance'])
#         interpolated_peaks = pk.interpolate(np.arange(0, len(cb)), cb, ind=peaks, width=3)

#         if len(peaks) == 0:
#             print('No peaks')
#             pass

#         peaks = peaks[:self.peak['num_peaks']]
#         self.axes[2].scatter(peaks, cb[peaks])
#         self.interpolated_peaks = interpolated_peaks[:self.peak['num_peaks']]

#         print('Peaks found:')
#         for i, p in enumerate(peaks):
#             print('-', i,  p, '', interpolated_peaks[i])

#         self._update()

#     def dp_deconv(self, event):
#         L, I, J, D = self.all_rirs.shape

#         # take the dp from... anechoic
#         a = self.dp_extreme[0]
#         b = self.dp_extreme[1]
#         dp = self.all_rirs[a:b, self.i, self.j, 0]
#         p = np.argmax(np.abs(dp))

#         print('Deconvolving with DP with dp in [%d, %d]' % (a, b))
#         for d in range(D):
#             rir = self.all_rirs[:, self.i, self.j, d]
#             dp_deconv = np.real(np.fft.ifft(np.fft.fft(rir, L) / np.fft.fft(dp, L)))[:L-p]
#             # restore the direct path
#             dp_deconv = np.concatenate([np.zeros(p), dp_deconv])
#             self.all_rirs_clean[:, self.i, self.j, d] = dp_deconv
#         print('Done')
#         self._update()



#     def _update(self):
#         xlim = self.axes[0].get_xlim()
#         ylim = self.axes[0].get_ylim()

#         print('Updating')
#         print(self.i, self.j)
#         # update current rirs
#         self.curr_rirs = self.all_rirs[:, self.i, self.j, :]
#         self.curr_rirs_clean = self.all_rirs_clean[:, self.i, self.j, :]

#         ## UPDATE PLOTS
#         # clear axes
#         self.axes[0].clear()
#         self.axes[1].clear()
#         self.axes[2].clear()
#         self.axes[3].clear()

#         self.axes[0].set_xlim(xlim)
#         self.axes[0].set_ylim(ylim)

#         plot_rirs_and_note(self.curr_rirs, self.toa_note, self.i, self.j, self.idx, self.axes[0], self.visibility, self.dp_extreme)
#         plot_rirs_and_note(self.curr_rirs_clean, self.toa_note, self.i, self.j, self.idx, self.axes[1], self.visibility, self.dp_extreme)

#         self.feature = plot_rirs_and_note_merged(
#             self.curr_rirs_clean, self.toa_note, self.i, self.j, self.idx, self.axes[2])

#         plot_staked_rirs(self.all_rirs, self.toa_note, self.i, self.j, self.idx, self.axes[3])

#         if not self.interpolated_peaks is None:
#             plot_peak_annotation(self.interpolated_peaks, self.feature, self.axes[2])

#         self.fig.canvas.draw_idle()


# # init
# axes = [ax1, ax2, ax3, ax4]
# cb = Callbacks(cp(toa_note), all_rirs, cp(all_rirs), datasets, initial_validity, fig, axes=axes)

# check.on_clicked(cb.set_visibility)

# dp_box1.on_submit(lambda string: cb.set_extreme_dp_deconvolution(int(string), 0))
# dp_box2.on_submit(lambda string: cb.set_extreme_dp_deconvolution(int(string), 1))
# dp_box.on_clicked(cb.dp_deconv)

# bt_save_tau.on_clicked(cb.save_tau_to_file)
# bt_load_tau.on_clicked(cb.load_tau_note)
# bt_peak_tau.on_clicked(cb.peak_peaking)

# tx_text_mic.on_submit(lambda i: cb.set_mic(int(i)))
# tx_text_src.on_submit(lambda j: cb.set_src(int(j)))

# txt_boxes_tau[0].on_submit(lambda tau_str: cb.set_tau_val(float(tau_str), 0))
# txt_boxes_tau[1].on_submit(lambda tau_str: cb.set_tau_val(float(tau_str), 1))
# txt_boxes_tau[2].on_submit(lambda tau_str: cb.set_tau_val(float(tau_str), 2))
# txt_boxes_tau[3].on_submit(lambda tau_str: cb.set_tau_val(float(tau_str), 3))
# txt_boxes_tau[4].on_submit(lambda tau_str: cb.set_tau_val(float(tau_str), 4))
# txt_boxes_tau[5].on_submit(lambda tau_str: cb.set_tau_val(float(tau_str), 5))
# txt_boxes_tau[6].on_submit(lambda tau_str: cb.set_tau_val(float(tau_str), 6))


# txt_boxes_pck_dst.on_submit(lambda val_str: cb.set_peak_dict('distance', float(val_str)))
# txt_boxes_pck_thr.on_submit(lambda val_str: cb.set_peak_dict('threshold', float(val_str)))
# txt_boxes_pck_num.on_submit(lambda val_str: cb.set_peak_dict('num_peaks', int(val_str)))

plt.show()
