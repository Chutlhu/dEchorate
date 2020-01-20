import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyroomacoustics as pra

# load positions
path_to_positions = './data/raw/positions.csv'
pos = pd.read_csv(path_to_positions)

mic_bar_pos = pos.loc[pos['type'] == 'array']
mic_theta = np.array(mic_bar_pos['theta'])
mic_bar_pos = np.vstack([mic_bar_pos['x'], mic_bar_pos['y'], mic_bar_pos['z']])

I = 5 * mic_bar_pos.shape[-1]

src_omni_pos = pos.loc[pos['type'] == 'omni']
src_omni_pos = np.vstack([src_omni_pos['x'], src_omni_pos['y'], src_omni_pos['z']])

src_dir_pos = pos.loc[pos['type'] == 'dir']
src_dir_pos = np.vstack([src_dir_pos['x'], src_dir_pos['y'], src_dir_pos['z']])

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
print(mics)

plt.gca().add_patch(
    plt.Rectangle((0, 0),
                   room_size[0], room_size[1], fill=False,
                   edgecolor='g', linewidth=1)
)

for i in range(I):
    plt.scatter(mics[0, i], mics[1, i], marker='X')
    plt.text(mics[0, i], mics[1, i], '$%d$' %
             (i+33), fontdict={'fontsize': 8})
    # if i % 5 == 0:
    #     bar = np.mean(mics[:2, 5*i:5*(i+1)], axis=1)
    #     plt.text(bar[0]+0.2, bar[1]+0.2, '$arr_%d$ [%1.2f, %1.2f]' %
    #              (i % 5+1, bar[0], bar[1]))


# for j in range(J):
#     bar = srcs[:, j]
#     if j < 4:
#         plt.scatter(bar[0], bar[1], marker='v')
#         plt.text(bar[0], bar[1], '$dir_%d$ [%1.2f, %1.2f, %1.2f]' %
#                  (j+1, bar[0], bar[1], bar[2]))
#     else:
#         plt.scatter(bar[0], bar[1], marker='o')
#         plt.text(bar[0], bar[1], '$omn_%d$ [%1.2f, %1.2f, %1.2f]' %
#                  (j+1, bar[0], bar[1], bar[2]))
plt.show()



room = pra.ShoeBox(room_size)

