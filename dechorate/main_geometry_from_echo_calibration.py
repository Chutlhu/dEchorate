import numpy as np
import pandas as pd
import pyroomacoustics as pra

import matplotlib.pyplot as plt

from utils.file_utils import save_to_pickle

from pathlib import Path



if __name__ == "__main__":

    output_dir = Path('.','outputs')

    path_to_output_pos_csv = output_dir / Path('dEchorate_calibrated_elements_positions.csv')
    path_to_output_echo_pkl = output_dir / Path('dEchorate_echo_notes.pkl')

    ###############################################################################
    room_temperature = 24
    speed_of_sound = 331.3 + 0.606 * room_temperature

    ## ROOM SIZE
    room_size = [5.705, 5.965, 2.355]  # meters

    ## MICROPHONES :: calibrated
    mics = np.array(
    [[  0.80316092, 0.8406819,  0.88758314, 0.94855474, 1.0423572,  0.88136256,
        0.86610109, 0.84702425, 0.82222436, 0.78407068, 2.21190695, 2.24779803,
        2.29266187, 2.35098486, 2.44071254, 3.35956993, 3.3818327,  3.40966117,
        3.44583818, 3.50149512, 3.72758061, 3.70088482, 3.6675151,  3.62413445,
        3.55739499, 3.08212703, 3.04939035, 3.00846949, 2.95527238, 2.87343067],
    [   3.83141445, 3.84527719, 3.86260561, 3.88513257, 3.91978942, 2.18763190,
        2.15065775, 2.10444007, 2.04435708, 1.95192172, 1.71556362, 1.69790489,
        1.67583147, 1.64713603, 1.6029892,  2.40748639, 2.44071843, 2.48225849,
        2.53626056, 2.61934068, 4.02235716, 3.99256898, 3.95533377, 3.90692799,
        3.83245756, 3.42108318, 3.44406817, 3.4727994,  3.51015,    3.56761246],
    [   1.04391528, 1.04391528, 1.04391528, 1.04391528, 1.04391528, 0.98664873,
        0.98664873, 0.98664873, 0.98664873, 0.98664873, 1.30742631, 1.30742631,
        1.30742631, 1.30742631, 1.30742631, 1.38561103, 1.38561103, 1.38561103,
        1.38561103, 1.38561103, 0.95058622, 0.95058622, 0.95058622, 0.95058622,
        0.95058622, 1.49048013, 1.49048013, 1.49048013, 1.49048013, 1.49048013]]
    ) # meters [3xI]


    ## ARRAYS CENTERS :: calibrated
    arrs = np.array(
        [[  0.90446758, 0.84015659, 2.30881285, 3.41967942, 3.65550199, 2.99373799],
        [   3.86884385, 2.08780171, 1.66788504, 2.49721291, 3.94192909, 3.48314264],
        [   1.04391528, 0.98664873, 1.30742631, 1.38561103, 0.95058622, 1.49048013]]
    ) # meters [3xA]

    ## DIRECTIONAL SOURCES :: calibrated
    srcs_dir = np.array(
        [[  1.89407487, 1.63305090, 4.30997826, 4.61104211, 1.63305090, 4.46997826],
        [   4.52190448, 0.84200410, 2.14089470, 3.74513990, 0.68200410, 2.14089470],
        [   1.44818390, 1.02493109, 1.09632443, 1.50903224, 1.16493109, 1.23632443]]
    ) # meters [3xJ]
    srcs_dir_aim = np.array(
        ['s','n','w','w','s','e']
    )

    ## OMNIDIRECTIONAL SOURCES :: NOT calibrated ::
    srcs_omn = np.array(
        [[3.651, 2.958, 0.892],
         [1.004, 4.558, 3.013],
         [1.380, 1.486, 1.403]]
    ) # meters [3xJ]
    srcs = np.concatenate([srcs_dir, srcs_omn], axis=-1)

    ## BABBLE NOISE SOURCES :: NOT calibrated ::
    srcs_nse = np.array(
        [[0.960, 0.876, 4.644, 4.692],
         [4.769, 0.868, 0.812, 4.735],
         [1.437, 1.430, 1.408, 1.429]]
    ) # meters [3xJ]

    ###############################################################################
    ## BUILD THE DATABASE
    df = pd.DataFrame()
    c = 0

    # directional source
    for j in range(srcs_dir.shape[1]):
        df.at[c, 'id'] = int(j+1)
        df.at[c, 'type'] = 'directional'
        df.at[c, 'channel'] = int(33+j)
        df.at[c, 'x'] = srcs_dir[0, j]
        df.at[c, 'y'] = srcs_dir[1, j]
        df.at[c, 'z'] = srcs_dir[2, j]
        df.at[c, 'aiming_at'] = srcs_dir_aim[j]
        c += 1

    # omndirectional source
    for j in range(srcs_omn.shape[1]):
        df.at[c, 'id'] = int(j+1) + 6
        df.at[c, 'type'] = 'omnidirectional'
        df.at[c, 'channel'] = 16
        df.at[c, 'x'] = srcs_omn[0, j]
        df.at[c, 'y'] = srcs_omn[1, j]
        df.at[c, 'z'] = srcs_omn[2, j]
        c += 1

    for j in range(srcs_nse.shape[1]):
        df.at[c, 'id'] = int(j+1)
        df.at[c, 'type'] = 'diffuse'
        df.at[c, 'channel'] = int(45+j)
        df.at[c, 'x'] = srcs_nse[0, j]
        df.at[c, 'y'] = srcs_nse[1, j]
        df.at[c, 'z'] = srcs_nse[2, j]
        c += 1

    for i in range(mics.shape[1]):
        df.at[c, 'channel'] = int(33 + i)
        df.at[c, 'id'] = int(i+1)
        df.at[c, 'type'] = 'mic'
        df.at[c, 'array'] = int(i//5 + 1)
        df.at[c, 'x'] = mics[0, i]
        df.at[c, 'y'] = mics[1, i]
        df.at[c, 'z'] = mics[2, i]
        c += 1

    for i in range(arrs.shape[1]):
        df.at[c, 'id'] = int(i+1)
        df.at[c, 'type'] = 'array'
        df.at[c, 'x'] = arrs[0, i]
        df.at[c, 'y'] = arrs[1, i]
        df.at[c, 'z'] = arrs[2, i]
        c += 1

    df.to_csv(path_to_output_pos_csv)

    ###############################################################################
    ## PRINT FIGURES
    # Blueprint 2D xy plane
    marker_size = 120
    plt.rcParams.update({'font.size': 13})

    m = { # marker type
        'arrs' : 'x',
        'mics' : 'X',
        'srcs_dir' : 'v',
        'srcs_omn' : 'o',
        'srcs_nse' : 'D',
    }
    s = { # marker size
        'arrs' : 120,
        'mics' : 80,
        'srcs_dir' : 120,
        'srcs_omn' : 120,
        'srcs_nse' : 100,
    }
    c = { # colors
        'arrs' : 'k',
        'mics' : 'C0',
        'srcs_dir' : 'C1',
        'srcs_omn' : 'C2',
        'srcs_nse' : 'C3',
        
    }
    l = { # labels
        'arrs' : 'array barycenters',
        'mics' : 'microphones',
        'srcs_dir' : 'directional sources',
        'srcs_omn' : 'omnidirectional sources',
        'srcs_nse' : 'diffuse noise',
    }

    plt.figure(figsize=(8,8))

    # Plot ROOM
    plt.gca().add_patch(
        plt.Rectangle((0, 0),
                    room_size[0], room_size[1], fill=False,
                    edgecolor='g', linewidth=4)
    )

    plt.scatter(mics[0, :], mics[1, :], marker=m['mics'], s=s['mics'], c=c['mics'], label=l['mics'])
    plt.scatter(arrs[0, :], arrs[1, :], marker=m['arrs'], s=s['arrs'], c=c['arrs'], label=l['arrs'])

    plt.text(arrs[0, 0]+0.1, arrs[1, 0]-0.15, '$arr_%d$' %1)
    plt.text(arrs[0, 1]+0.1, arrs[1, 1]-0.15, '$arr_%d$' %2)
    plt.text(arrs[0, 2]+0.1, arrs[1, 2]+0.10, '$arr_%d$' %3)
    plt.text(arrs[0, 3]+0.1, arrs[1, 3]-0.15, '$arr_%d$' %4)
    plt.text(arrs[0, 4]+0.1, arrs[1, 4]-0.1, '$arr_%d$'  %5)
    plt.text(arrs[0, 5]+0.1, arrs[1, 5]+0.1, '$arr_%d$'  %6)

    # DIR
    plt.scatter(srcs_dir[0, 0], srcs_dir[1, 0], marker='v', s=s['srcs_dir'], c=c['srcs_dir'], label=l['srcs_dir'])
    plt.text(srcs_dir[0, 0]+0.1, srcs_dir[1, 0]-0.1, r'$dir_%d$' %0)
    plt.text(srcs_dir[0, 1]+0.1, srcs_dir[1, 1]+0.1, r'$dir_%d$' %1)
    plt.text(srcs_dir[0, 2]-0.2, srcs_dir[1, 2]-0.2, r'$dir_%d$' %2)
    plt.text(srcs_dir[0, 3]-0.2, srcs_dir[1, 3]-0.2, r'$dir_%d$' %3)
    plt.text(srcs_dir[0, 4]+0.1, srcs_dir[1, 4]-0.1, r'$dir_%d$' %4)
    plt.text(srcs_dir[0, 5]+0.1, srcs_dir[1, 5]-0.2, r'$dir_%d$' %5)

    # DIR
    plt.scatter(srcs_dir[0, 1], srcs_dir[1, 1], marker='^', s=s['srcs_dir'], c=c['srcs_dir'])
    plt.scatter(srcs_dir[0, 2], srcs_dir[1, 2], marker='<', s=s['srcs_dir'], c=c['srcs_dir'])
    plt.scatter(srcs_dir[0, 3], srcs_dir[1, 3], marker='<', s=s['srcs_dir'], c=c['srcs_dir'])
    plt.scatter(srcs_dir[0, 4], srcs_dir[1, 4], marker='v', s=s['srcs_dir'], c=c['srcs_dir'])
    plt.scatter(srcs_dir[0, 5], srcs_dir[1, 5], marker='>', s=s['srcs_dir'], c=c['srcs_dir'])

    # OMNI
    plt.scatter(srcs_omn[0, :], srcs_omn[1, :], marker=m['srcs_omn'], s=s['srcs_omn'], c=c['srcs_omn'], label=l['srcs_omn'])
    plt.text(srcs_omn[0, 0]-0.2, srcs_omn[1, 0]-0.2, r'$omni_%d$' %0)
    plt.text(srcs_omn[0, 1]-0.2, srcs_omn[1, 1]-0.2, r'$omni_%d$' %1)
    plt.text(srcs_omn[0, 2]-0.2, srcs_omn[1, 2]-0.2, r'$omni_%d$' %2)


    # NOISE
    plt.scatter(srcs_nse[0, :], srcs_nse[1, :], marker=m['srcs_nse'], s=s['srcs_nse'], c=c['srcs_nse'], label=l['srcs_nse'])
    plt.text(srcs_nse[0, 0]-0.2, srcs_nse[1, 0]-0.25, r'$noise_%d$' %0)
    plt.text(srcs_nse[0, 1]-0.2, srcs_nse[1, 1]-0.25, r'$noise_%d$' %1)
    plt.text(srcs_nse[0, 2]-0.2, srcs_nse[1, 2]-0.25, r'$noise_%d$' %2)
    plt.text(srcs_nse[0, 3]-0.2, srcs_nse[1, 3]-0.25, r'$noise_%d$' %3)


    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / Path('positioning2D_xy.pdf'))
    # plt.show()
    plt.close()

    ###############################################################################

    # Create the room
    room = pra.ShoeBox(
        room_size, fs=16000, max_order=2
    )

    # place the mics in the room
    room.add_microphone_array(mics)

    # place the source in teh room
    for j in range(srcs_dir.shape[1]):
        room.add_source(position=srcs_dir[:,j])

    # run Image Source model
    room.image_source_model()
    room.compute_rir()

    # 2nd order reflection => 25 images
    K = 25
    I = mics.shape[1]
    J = srcs_dir.shape[1]

    echoes_toa = np.zeros((K, I, J))
    echoes_amp = np.zeros((K, I, J))

    for i in range(mics.shape[1]):
        for j in range(srcs_dir.shape[1]):
            
            source = room.sources[j]
            images_pos = source.get_images(max_order=2)
            images_damp = source.get_damping(max_order=2)
        
            images_dist = np.linalg.norm(images_pos - mics[:,i,None], axis=0)
        
            toas = (images_dist / speed_of_sound).squeeze()
            amps = (images_damp / (4*np.pi*images_dist)).squeeze()

            idx = np.argsort(toas)

            echoes_toa[:,i,j] = toas[idx]
            echoes_amp[:,i,j] = amps[idx]

    echo_note = {
        'mics' : mics,
        'srcs' : srcs_dir,
        'toa' : echoes_toa,
        'amp' : echoes_amp
    }

    save_to_pickle(path_to_output_echo_pkl, echo_note)
    print('Echo notes saved in', path_to_output_echo_pkl)