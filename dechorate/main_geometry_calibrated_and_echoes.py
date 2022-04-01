import numpy as np
import pandas as pd
import pyroomacoustics as pra

import matplotlib.pyplot as plt

from pathlib import Path

###############################################################################
# optional
# # Load annotation from csv
# positions_note = pd.read_csv('../../Datasets/dEchorate/dECHORATE_positioning_annotation.csv')
# mics = positions_note.loc[positions_note['type'] == 'mic'][['x', 'y', 'z']].values.T
# srcs_dir = positions_note.loc[positions_note['type'] == 'directional'][['x', 'y', 'z']].values.T
# srcs_omn = positions_note.loc[positions_note['type'] == 'omnidirectional'][['x', 'y', 'z']].values.T
# srcs_nse = positions_note.loc[positions_note['type'] == 'babble noise'][['x', 'y', 'z']].values.T
# arrs = positions_note.loc[positions_note['type'] == 'array'][['x', 'y', 'z']].values.T

###############################################################################
room_temperature = 24
speed_of_sound = 331.3 + 0.606 * room_temperature

## ROOM SIZE
room_size = [5.705, 5.965, 2.355]  # meters

## MICROPHONES
mics = np.array(
[[0.85771319, 0.89503133, 0.94167901, 1.00232099, 1.09561634, 0.81689747,
  0.80321666, 0.78611565, 0.76388435, 0.72968233, 2.18273871, 2.21808934,
  2.26227762, 2.31972238, 2.40809895, 3.34254184, 3.36391593, 3.39063355,
  3.42536645, 3.47880169, 3.77147839, 3.74389361, 3.70941264, 3.66458736,
  3.59562541, 3.12959949, 3.09740374, 3.05715905, 3.00484095, 2.92435157,],
 [3.90990039, 3.92430026, 3.9423001,  3.9656999,  4.00169958, 2.24411235,
  2.20652464, 2.15954001, 2.09845999, 2.00449073, 1.7263214,  1.70760421,
  1.68420772, 1.65379228, 1.6069993,  2.35745543, 2.3912659,  2.43352899,
  2.48847101, 2.57299719, 3.96471105, 3.93574418, 3.89953559, 3.85246441,
  3.78004723, 3.37230619, 3.39604294, 3.42571389, 3.46428611, 3.523628,  ],
 [1.039,      1.039,      1.039,      1.039,      1.039,      0.98,
  0.98 ,      0.98,       0.98,       0.98,       1.163,      1.163,
  1.163,      1.163,      1.163,      1.31,       1.31,       1.31,
  1.31 ,      1.31,       0.91,       0.91,       0.91,       0.91,
  0.91 ,      1.46,       1.46,       1.46,       1.46,       1.46,      ]]
) # meters [3xI]

## ARRAYS CENTERS
arrs = np.array(
[[0.972, 0.775, 2.291, 3.408, 3.687, 3.031],
 [3.954, 2.129, 1.669, 2.461, 3.876, 3.445],
 [1.039, 0.98,  1.163, 1.31,  0.91,  1.46]]
) # meters [3xA]

## DIRECTIONAL SOURCES
srcs_dir = np.array(
[[1.991, 1.523, 4.287, 4.619, 1.523, 4.447],
 [4.498, 0.883, 2.069, 3.669, 0.723, 2.069],
 [1.424, 1.044, 1.074, 1.504, 1.184, 1.214]]
) # meters [3xJ]
srcs_dir_aim = np.array(
    ['s','n','w','w','s','e']
)

## OMNIDIRECTIONAL SOURCES
srcs_omn = np.array(
[[3.651, 2.958, 0.892],
 [1.004, 4.558, 3.013],
 [1.38 , 1.486, 1.403]]
) # meters [3xJ]

## BABBLE NOISE SOURCES
srcs_nse = np.array(
[[0.96 , 0.876, 4.644, 4.692],
 [4.769, 0.868, 0.812, 4.735],
 [1.437, 1.43,  1.408, 1.429]]
) # meters [3xJ]

###############################################################################
## BUILD THE DATABASE
df = pd.DataFrame()
c = 0

# directional source
for j in range(srcs_dir.shape[1]):
    df.at[c, 'id'] = int(j)+1
    df.at[c, 'type'] = 'directional'
    df.at[c, 'channel'] = int(33+j)
    df.at[c, 'x'] = srcs_dir[0, j]
    df.at[c, 'y'] = srcs_dir[1, j]
    df.at[c, 'z'] = srcs_dir[2, j]
    df.at[c, 'aiming_at'] = srcs_dir_aim[j]
    c += 1

# omndirectional source
for j in range(srcs_omn.shape[1]):
    df.at[c, 'id'] = int(j)+1
    df.at[c, 'type'] = 'omnidirectional'
    df.at[c, 'channel'] = 16
    df.at[c, 'x'] = srcs_omn[0, j]
    df.at[c, 'y'] = srcs_omn[1, j]
    df.at[c, 'z'] = srcs_omn[2, j]
    c += 1

for j in range(srcs_nse.shape[1]):
    df.at[c, 'id'] = int(j)+1
    df.at[c, 'type'] = 'noise'
    df.at[c, 'channel'] = int(45+j)
    df.at[c, 'x'] = srcs_nse[0, j]
    df.at[c, 'y'] = srcs_nse[1, j]
    df.at[c, 'z'] = srcs_nse[2, j]
    c += 1

for i in range(mics.shape[1]):
    df.at[c, 'channel'] = int(33 + i)
    df.at[c, 'id'] = int(i)+1
    df.at[c, 'type'] = 'mic'
    df.at[c, 'array'] = int(i//5 + 1)
    df.at[c, 'x'] = mics[0, i]
    df.at[c, 'y'] = mics[1, i]
    df.at[c, 'z'] = mics[2, i]
    c += 1

for i in range(arrs.shape[1]):
    df.at[c, 'id'] = int(i)
    df.at[c, 'type'] = 'array'
    df.at[c, 'x'] = arrs[0, i]
    df.at[c, 'y'] = arrs[1, i]
    df.at[c, 'z'] = arrs[2, i]
    c += 1

df.to_csv(Path('elements_positions.csv'))

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

plt.text(arrs[0, 0]+0.1, arrs[1, 0]-0.15, '$arr_%d$' %0)
plt.text(arrs[0, 1]+0.1, arrs[1, 1]-0.15, '$arr_%d$' %1)
plt.text(arrs[0, 2]+0.1, arrs[1, 2]+0.10, '$arr_%d$' %2)
plt.text(arrs[0, 3]+0.1, arrs[1, 3]-0.15, '$arr_%d$' %3)
plt.text(arrs[0, 4]+0.1, arrs[1, 4]-0.1, '$arr_%d$' %4)
plt.text(arrs[0, 5]+0.1, arrs[1, 5]+0.1, '$arr_%d$' %5)

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
plt.savefig('positioning2D_xy.pdf')
plt.show()

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
       
        echoes_toa[:,i,j] = images_dist / speed_of_sound
        echoes_amp[:,i,j] = images_damp / (4*np.pi*images_dist)