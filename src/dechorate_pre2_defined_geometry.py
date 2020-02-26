import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import pyroomacoustics as pra

from src.utils.file_utils import load_from_matlab

## LOAD POSITIONS
# load arrays' barycenters positions from file generated in excell
# using data from the beacon and sketchUp (for theta)
path_to_positions = './data/dECHORATE/positions.csv'
audio_scene_positions = pd.read_csv(path_to_positions)

mic_bar_pos = audio_scene_positions.loc[audio_scene_positions['type'] == 'array']
mic_theta = np.array(mic_bar_pos['theta'])
mic_aim = np.array(mic_bar_pos['aiming_at'])
mic_bar_pos = np.vstack([mic_bar_pos['x'], mic_bar_pos['y'], mic_bar_pos['z']])

I = 5 * mic_bar_pos.shape[-1]

src_omni_pos = audio_scene_positions.loc[audio_scene_positions['type'] == 'omni']
src_omni_pos = np.vstack([src_omni_pos['x'], src_omni_pos['y'], src_omni_pos['z']])

src_noise_pos = audio_scene_positions.loc[audio_scene_positions['type'] == 'noise']
src_noise_pos = np.vstack([src_noise_pos['x'], src_noise_pos['y'], src_noise_pos['z']])

src_dir_pos = audio_scene_positions.loc[audio_scene_positions['type'] == 'dir']
src_dir_pos = np.vstack([src_dir_pos['x'], src_dir_pos['y'], src_dir_pos['z']])
Jd = src_dir_pos.shape[-1]
Jo = src_omni_pos.shape[-1]
Jn = src_noise_pos.shape[-1]
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

# built sources
srcs = np.zeros([3, J])
# directional sound sources
srcs[:, :Jd] = src_dir_pos
# omnidirectional sound sources
srcs[:, Jd:] = src_omni_pos

## BUILD THE DATABASE
df = pd.DataFrame()

# directional source
c = 0
for j in range(Jd):
    df.at[c, 'id'] = int(j)+1
    df.at[c, 'type'] = 'directional'
    df.at[c, 'channel'] = int(33+j)
    c += 1

# omndirectional source
c = len(df)
for j in range(Jo):
    df.at[c, 'id'] = int(j)+7
    df.at[c, 'type'] = 'omnidirectional'
    df.at[c, 'channel'] = 16
    if j == 0:
        df.at[c, 'x'] = srcs[0, 4]
        df.at[c, 'y'] = srcs[1, 4]
        df.at[c, 'z'] = srcs[2, 4]
    if j == 1:
        df.at[c, 'x'] = srcs[0, 6]
        df.at[c, 'y'] = srcs[1, 6]
        df.at[c, 'z'] = srcs[2, 6]
    if j == 2:
        df.at[c, 'x'] = srcs[0, 5]
        df.at[c, 'y'] = srcs[1, 5]
        df.at[c, 'z'] = srcs[2, 5]
    c += 1

c = len(df)
for j in range(Jn):
    df.at[c, 'id'] = int(j)+1
    df.at[c, 'type'] = 'babble noise'
    df.at[c, 'channel'] = int(45+j)
    df.at[c, 'x'] = src_noise_pos[0, j]
    df.at[c, 'y'] = src_noise_pos[1, j]
    df.at[c, 'z'] = src_noise_pos[2, j]
    c += 1

c = len(df)
for i in range(I):
    df.at[c, 'channel'] = int(33 + i)
    df.at[c, 'id'] = int(i)+1
    df.at[c, 'type'] = 'mic'
    df.at[c, 'array'] = int(i//5 + 1)
    df.at[c, 'x'] = mics[0, i]
    df.at[c, 'y'] = mics[1, i]
    df.at[c, 'z'] = mics[2, i]
    c += 1

c = len(df)
for i in range(I//5):
    df.at[c, 'channel'] = int(33 + i*5)
    df.at[c, 'id'] = int(i)+1
    df.at[c, 'type'] = 'array'
    df.at[c, 'array'] = i+1
    df.at[c, 'theta'] = mic_theta[i]
    df.at[c, 'aiming_at'] = mic_aim[i]
    df.at[c, 'x'] = mic_bar_pos[0, i]
    df.at[c, 'y'] = mic_bar_pos[1, i]
    df.at[c, 'z'] = mic_bar_pos[2, i]
    c += 1

csv_filename = './data/dECHORATE/annotations/dECHORATE_positioning_note.csv'
df.to_csv(csv_filename, sep=',')

## PRINT FIGURES
# Blueprint 2D
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
plt.savefig('./data/dECHORATE/pictures/positioning2D.pdf')
plt.show()

# plot with pyroomacoustics
room = pra.ShoeBox(room_size, fs=48000)

room.add_microphone_array(pra.MicrophoneArray(mics, room.fs))
for j in range(J):
    room.add_source(srcs[:, j])
plt.savefig('./data/dECHORATE/pictures/positioning3D.pdf')
room.plot()
plt.show()


