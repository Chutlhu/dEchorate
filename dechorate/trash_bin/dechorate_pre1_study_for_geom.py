import numpy as np
import scipy as sp
import pandas as pd
import pyroomacoustics as pra
import matplotlib.pyplot as plt

room_size = [5.741, 5.763, 2.353]

room = pra.ShoeBox(room_size)

min_dist = 0.71
max_dist = 5.741/2

step_distances = (max_dist - min_dist)/6

distances_from_walls = np.array([min_dist + i*step_distances for i in range(6)])

mic0_center = np.array(
    [distances_from_walls[1], 3.5*distances_from_walls[1] + 0.2])
mic0 = pra.linear_2D_array(mic0_center, 5, 45, 0.075)

mic1_center = np.array([distances_from_walls[0], 3*distances_from_walls[0]])
mic1 = pra.linear_2D_array(mic1_center, 5, 30, 0.075)

mic2_center = np.array(
    [1.3*distances_from_walls[3], distances_from_walls[3]])
mic2 = pra.linear_2D_array(mic2_center, 5, 20, 0.075)

mic3_center = np.array(
    [room_size[0]-distances_from_walls[4], 1.2*distances_from_walls[4]])
mic3 = pra.linear_2D_array(mic3_center, 5, -30, 0.075)

mic4_center = np.array(
    [room_size[0]-1.5*distances_from_walls[2]-0.2, room_size[1]-distances_from_walls[2]])
mic4 = pra.linear_2D_array(mic4_center, 5, 60, 0.075)


mic5_center = np.array(room_size[:2])/2
mic5_center = np.array([3, 3.5])
mic5 = pra.linear_2D_array(mic5_center, 5, 15, 0.075)


mics = np.zeros([3, 30])
mics[:2,  0:5]  = mic0
mics[:2,  5:10] = mic1
mics[:2, 10:15] = mic2
mics[:2, 15:20] = mic3
mics[:2, 20:25] = mic4
mics[:2, 25:30] = mic5
mics[2,   0:5] = 1.5
mics[2,  5:10] = 0.9
mics[2, 10:15] = 1.3
mics[2, 15:20] = 1.5
mics[2, 20:25] = 1.2
mics[2, 25:30] = 1.5
room.add_microphone_array(pra.MicrophoneArray(mics, room.fs))
print('Mics position')
for i in range(mics.shape[-1]):
    print(mics[:, i])

src1 = [1.5, 5, 1.7]
src2 = [1.5, 1, 1.2]
src3 = [5, 1.5, 1.]
src4 = [5, 4, 1.5]

src5 = [2.5, 4, 1.5]
src6 = [1, 3, 1.5]
src7 = [4, 1, 1.5]


room.add_source(src1)
room.add_source(src2)
room.add_source(src3)

room.plot()
plt.show()


room.add_source(src4)
room.add_source(src5)
room.add_source(src6)
room.add_source(src7)

srcs = np.zeros([3, 7])
srcs[:, 0] = np.array(src1)
srcs[:, 1] = np.array(src2)
srcs[:, 2] = np.array(src3)
srcs[:, 3] = np.array(src4)
srcs[:, 4] = np.array(src5)
srcs[:, 5] = np.array(src6)
srcs[:, 6] = np.array(src7)


I = 6
J = 7

plt.figure(figsize=(9,9))

plt.plot([0, 0], [room_size[0], 0], 'k')
plt.plot([room_size[0], 0], [room_size[0], room_size[1]], 'k')
plt.plot([room_size[0], room_size[1]], [0, room_size[1]], 'k')
plt.plot([0, room_size[1]], [0, 0], 'k')

for i in range(I):
    bar = np.mean(mics[:, 5*i:5*(i+1)], axis=1)
    plt.scatter(bar[0], bar[1], marker='X')
    plt.text(bar[0], bar[1], '$arr_%d$ [%1.2f, %1.2f, %1.2f]' % (i+1, bar[0], bar[1], bar[2]))

for j in range(J):
    bar = srcs[:, j]
    if j < 4:
        plt.scatter(bar[0], bar[1], marker='v')
        plt.text(bar[0], bar[1], '$dir_%d$ [%1.2f, %1.2f, %1.2f]' % (j+1, bar[0], bar[1], bar[2]))
    else:
        plt.scatter(bar[0], bar[1], marker='o')
        plt.text(bar[0], bar[1], '$omn_%d$ [%1.2f, %1.2f, %1.2f]' % (j+1, bar[0], bar[1], bar[2]))

plt.show()