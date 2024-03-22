import numpy as np
import pandas as pd
import argparse

import matplotlib.pyplot as plt
import pyroomacoustics as pra

from dechorate import constants

from pathlib import Path

room_size = constants["room_size"]

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", help="Path to output files", type=str)
    parser.add_argument("--datadir", help="Path to dEchorate raw dataset", type=str)
    args = parser.parse_args()

    data_dir = args.datadir

    output_dir = Path(args.outdir)

    ## LOAD POSITIONS
    # load arrays' barycenters positions from file generated in excell
    # using data from the beacon and sketchUp (for theta)
    path_to_positions = data_dir / Path('dEchorate_positions_marvel.csv')
    audio_scene_positions = pd.read_csv(path_to_positions)

    mic_bar_pos = audio_scene_positions.loc[audio_scene_positions['type'] == 'array']
    mic_theta = np.array(mic_bar_pos['theta'])
    mic_aim = np.array(mic_bar_pos['aiming_at'])
    mic_bar_pos = np.vstack([mic_bar_pos['x'], mic_bar_pos['y'], room_size[2] + mic_bar_pos['z']])

    I = 5 * mic_bar_pos.shape[-1]

    src_omni_pos = audio_scene_positions.loc[audio_scene_positions['type'] == 'omni']
    src_omni_pos = np.vstack([src_omni_pos['x'], src_omni_pos['y'], room_size[2] + src_omni_pos['z']])

    src_noise_pos = audio_scene_positions.loc[audio_scene_positions['type'] == 'noise']
    src_noise_pos = np.vstack([src_noise_pos['x'], src_noise_pos['y'], room_size[2] + src_noise_pos['z']])

    src_dir_pos = audio_scene_positions.loc[audio_scene_positions['type'] == 'dir']
    src_dir_aim = np.array(src_dir_pos['aiming_at'])
    src_dir_pos = np.vstack([src_dir_pos['x'], src_dir_pos['y'], room_size[2] + src_dir_pos['z']])

    Jd = src_dir_pos.shape[-1]
    Jo = src_omni_pos.shape[-1]
    Jn = src_noise_pos.shape[-1]
    J = Jd + Jo


    ## Create linear arrays
    nULA = np.zeros([3,5])
    nULA[0, :] = np.array([0-3.25-5-4, 0-3.25-5, 0-3.25, 3.25, 3.25+10])/100

    def rotate_and_translate(LA, new_center, new_angle):
        # azimuth rotation
        # can be extended as in http://planning.cs.uiuc.edu/node102.html
        # rotate
        th = np.deg2rad(new_angle)
        R = np.array([[[np.cos(th), -np.sin(th), 0],
                    [np.sin(th), np.cos(th),  0],
                    [0,          0,          1]]])
        nULA_rot = R @ LA
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
        df.at[c, 'x'] = srcs[0, j]
        df.at[c, 'y'] = srcs[1, j]
        df.at[c, 'z'] = srcs[2, j]
        df.at[c, 'aiming_at'] = src_dir_aim[j]
        c += 1

    # omndirectional source
    c = len(df)
    for j in range(Jd, Jd+Jo):
        df.at[c, 'id'] = int(j)+1
        df.at[c, 'type'] = 'omnidirectional'
        df.at[c, 'channel'] = 16
        df.at[c, 'x'] = srcs[0, j]
        df.at[c, 'y'] = srcs[1, j]
        df.at[c, 'z'] = srcs[2, j]
        c += 1

    c = len(df)
    for j in range(Jn):
        df.at[c, 'id'] = int(j)+1
        df.at[c, 'type'] = 'noise'
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
        df.at[c, 'array'] = i
        df.at[c, 'theta'] = mic_theta[i]
        df.at[c, 'aiming_at'] = mic_aim[i]
        df.at[c, 'x'] = mic_bar_pos[0, i]
        df.at[c, 'y'] = mic_bar_pos[1, i]
        df.at[c, 'z'] = mic_bar_pos[2, i]
        c += 1

    csv_filename = output_dir / Path('dECHORATE_positioning_marvel_expanded.csv')
    df.to_csv(csv_filename, sep=',')

    ## PRINT FIGURES
    # Blueprint 2D xy plane
    font_size = 22
    marker_size = 120
    figsize = (12,9)

    plt.figure(figsize=figsize)
    plt.gca().add_patch(
        plt.Rectangle((0, 0),
                    room_size[0], room_size[1], fill=False,
                    edgecolor='g', linewidth=2)
    )

    plt.scatter(mics[0, :], mics[1, :], marker='X', s=80, label='microphones')

    bars = np.zeros([3, 6])
    c = 0
    for i in range(I):
        
        if i % 5 == 0:
            bar = np.mean(mics[:, 5*i//5:5*(i//5+1)], axis=1)
            bars[:, c] = bar

            if i//5 == 1:
                plt.text(bar[0]+0.1, bar[1]-0.05, '$arr_%d$' %(i//5), fontdict={'fontsize': font_size})
            else:
                plt.text(bar[0], bar[1]-0.2, '$arr_%d$' % (i//5), fontdict={'fontsize': font_size})
            c += 1

    plt.scatter(bars[0, :], bars[1, :], marker='1', s=marker_size, c='k', label='array barycenters')

    plt.scatter(srcs[0, :Jd], srcs[1, :Jd], marker='v', s=marker_size, label='directional')

    for j in range(Jd):
        if j == 2:
            plt.text(srcs[0, j], srcs[1, j]+0.1, '$dir_%d$' % (j), fontdict={'fontsize': font_size})
        elif j == 5:
            plt.text(srcs[0, j], srcs[1, j]-0.15, '$dir_%d$' % (j), fontdict={'fontsize': font_size})
        else:
            plt.text(srcs[0, j], srcs[1, j], '$dir_%d$' % (j), fontdict={'fontsize': font_size})


    plt.scatter(srcs[0, Jd:], srcs[1, Jd:], marker='o', s=marker_size, label='omnidirectional')

    for j in range(Jd, J):
        plt.text(srcs[0, j], srcs[1, j], '$omni_%d$' % (j), fontdict={'fontsize': font_size})


    plt.scatter(src_noise_pos[0,:], src_noise_pos[1,:], marker='o', s=marker_size, label='noise')

    for j in range(Jn):
        plt.text(src_noise_pos[0, j], src_noise_pos[1, j], '$noise_%d$' % (j), fontdict={'fontsize': font_size})

    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / Path('positioning2D_xy.pdf'))
    # plt.show()
    plt.close()

    # plot with pyroomacoustics
    room = pra.ShoeBox(room_size, fs=48000)

    room.add_microphone_array(pra.MicrophoneArray(mics, room.fs))
    for j in range(J):
        try:
            room.add_source(srcs[:, j])
        except:
            print('src', j, srcs[:, j], 'is out of the room')
    room.plot()
    plt.savefig(output_dir / Path('positioning3D.pdf'))
    # plt.show()
    plt.close()


