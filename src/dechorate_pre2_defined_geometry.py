import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import pyroomacoustics as pra

from src.utils.file_utils import load_from_mat

## LOAD POSITIONS
# load arrays' barycenters positions from file generated in excell
# using data from dechorate_pre1_study_for_goem.py
path_to_positions = './data/raw/positions.csv'
audio_scene_positions = pd.read_csv(path_to_positions)

mic_bar_pos = audio_scene_positions.loc[audio_scene_positions['type'] == 'array']
mic_theta = np.array(mic_bar_pos['theta'])
mic_bar_pos = np.vstack([mic_bar_pos['x'], mic_bar_pos['y'], mic_bar_pos['z']])

I = 5 * mic_bar_pos.shape[-1]

src_omni_pos = audio_scene_positions.loc[audio_scene_positions['type'] == 'omni']
src_omni_pos = np.vstack([src_omni_pos['x'], src_omni_pos['y'], src_omni_pos['z']])

src_dir_pos = audio_scene_positions.loc[audio_scene_positions['type'] == 'dir']
src_dir_pos = np.vstack([src_dir_pos['x'], src_dir_pos['y'], src_dir_pos['z']])
Jd = src_dir_pos.shape[-1]
Jo = src_omni_pos.shape[-1]
J = Jd + Jo

room_size = [5.543, 5.675, 2.353]

## Create linear arrays
nULA = np.zeros([3,5])
nULA[0, :] = np.array([0-3.25-5-4, 0-3.25-5, 0-3.25, 3.25, 3.25+10])/100

def rotate_and_translate(LA, new_center, new_angle):
    # rotate
    th = np.deg2rad(new_angle)
    R = np.array([[[np.cos(th), -np.sin(th), 0],
                   [np.sin(th), np.cos(th),  0],
                   [0,          0,          1]]])
    nULA_rot = R@LA
    # translate
    nULA_tra = nULA_rot + new_center[:, None]
    return nULA_tra

mics = np.zeros([3, I])
mics[:, 0:5]   = rotate_and_translate(nULA, mic_bar_pos[:, 0], mic_theta[0])
mics[:, 5:10]  = rotate_and_translate(nULA, mic_bar_pos[:, 1], mic_theta[1])
mics[:, 10:15] = rotate_and_translate(nULA, mic_bar_pos[:, 2], mic_theta[2])
mics[:, 15:20] = rotate_and_translate(nULA, mic_bar_pos[:, 3], mic_theta[3])
mics[:, 20:25] = rotate_and_translate(nULA, mic_bar_pos[:, 4], mic_theta[4])
mics[:, 25:30] = rotate_and_translate(nULA, mic_bar_pos[:, 5], mic_theta[5])

srcs = np.zeros([3, J])
srcs[:, :Jd] = src_dir_pos
srcs[:, Jd:] = src_omni_pos

mics[2, :] += room_size[2]
srcs[2, :] += room_size[2]

## UPDATE THE DATABASE
c = len(audio_scene_positions)
for i in range(mics.shape[1]):
    audio_scene_positions.at[c, 'channel'] = int(33 + i)
    audio_scene_positions.at[c, 'id'] = int(i)
    audio_scene_positions.at[c, 'type'] = 'mic'
    audio_scene_positions.at[c, 'array'] = int(i//5 + 1)
    audio_scene_positions.at[c, 'x'] = mics[0, i]
    audio_scene_positions.at[c, 'y'] = mics[0, i]
    audio_scene_positions.at[c, 'z'] = mics[0, i]
    c += 1

csv_filename = './data/dECHORATE/annotations/dECHORATE_positioning_note.csv'
audio_scene_positions.to_csv(csv_filename, sep=',')

## PRINT FIGURES

plt.gca().add_patch(
    plt.Rectangle((0, 0),
                   room_size[0], room_size[1], fill=False,
                   edgecolor='g', linewidth=1)
)

for i in range(I):
    plt.scatter(mics[0, i], mics[1, i], marker='X')
    plt.text(mics[0, i], mics[1, i], '$%d$' %
             (i+33), fontdict={'fontsize': 8})
    if i % 5 == 0:
        bar = np.mean(mics[:, 5*i//5:5*(i//5+1)], axis=1)
        plt.text(bar[0]+0.1, bar[1]+0.1, '$arr_%d$ [%1.2f, %1.2f, %1.2f]' %
                 (i//5 + 1, bar[0], bar[1], bar[2]))


for j in range(J):
    bar = srcs[:, j]
    if j < Jd:
        plt.scatter(bar[0], bar[1], marker='v')
        plt.text(bar[0], bar[1], '$dir_%d$ [%1.2f, %1.2f, %1.2f]' %
                 (j+1, bar[0], bar[1], bar[2]))
    else:
        plt.scatter(bar[0], bar[1], marker='o')
        plt.text(bar[0], bar[1], '$omn_%d$ [%1.2f, %1.2f, %1.2f]' %
                 (j+1, bar[0], bar[1], bar[2]))
plt.savefig('./data/dECHORATE/pictures/positioning.pdf')
plt.show()

# Fs = 48000
# K = 3
# room = pra.ShoeBox(room_size, fs=48000, max_order=K)

# mics = mics[:, 15:20]
# mics[2, :] = 1.405

# room.add_microphone_array(pra.MicrophoneArray(mics, room.fs))
# room.add_source(list(srcs[:, 4]))

# room.image_source_model(use_libroom=False)

# # room.plot_rir()
# # plt.show()
# K = 20  # len(room.sources[0].damping)
# toa = np.zeros([I, J, K])
# amp = np.zeros([I, J, K])
# walls = np.zeros([I, J, K])
# order = np.zeros([I, J, K])
# for i in range(5):
#     for j in range(1):
#         images = room.sources[j].images
#         center = room.mic_array.center
#         distances = np.linalg.norm(
#             images - room.mic_array.R[:, i, None], axis=0)
#         # order in loc
#         ordering = np.argsort(distances)[:K]
#         for o, k in enumerate(ordering):
#             amp[i, j, o] = room.sources[j].damping[k] / \
#                 (4 * np.pi * distances[k])
#             toa[i, j, o] = distances[k]/340
#             walls[i, j, o] = room.sources[j].walls[k]
#             order[i, j, o] = room.sources[j].orders[k]

# h = load_from_mat('./data/recordings/omni/rir_omni.mat')['H']
# times = np.arange(h.shape[0])/Fs
# h = h[:, 0] / np.max(np.abs(h[:, 0]))
# h *= amp[0, 0, 0]

# plt.plot(times[6000:8720], np.abs(sp.signal.hilbert(h[6000:8720])))
# plt.stem(toa[0, 0, :]+0.1356, amp[0, 0, :])

# for k in range(K):
#     wall_idx = int(walls[0, 0, k])
#     if wall_idx == -1:
#         wall_name = 'direct'
#     else:
#         wall_name = '%s:%d' % (room.wall_names[wall_idx], order[0, 0, k])
#     plt.text(toa[0, 0, k]+0.1356, amp[0, 0, k], wall_name)
# plt.show()


# save geometry in database
# df = pd.DataFrame()
# c = 0
# for i in range(mics.shape[1]):
#     df.at[c, 'object'] = 'microphone'
#     df.at[c, 'position'] = mics[:, i]
#     df.at[c, 'channel'] = i + 33
#     df.at[c, 'id'] = i
#     df.at[c, 'array'] = i//5

#     c += 1

# for j in range(J):
#     df.at[c, 'object'] = 'sources'
#     df.at[c, 'position'] = srcs[:, i]
#     df.at[c, 'channel'] = j
#     df.at[c, 'id'] = j
#     if j < 4:
#         df.at[c, 'type'] = 'directional'
#     else:
#         df.at[c, 'type'] = 'omnidirectional'

#     c += 1

# print(df)
# 1/0
