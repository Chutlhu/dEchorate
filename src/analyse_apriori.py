import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt

room_size = [5.741, 5.763]

room = pra.ShoeBox(room_size)

min_dist = 0.71
max_dist = 5.741/2



step_distances = (max_dist - min_dist)/6

distances_from_walls = np.array([min_dist + i*step_distances for i in range(6)])
print(distances_from_walls)

k = 6
mic0_center = np.array(room_size)/2
mic0 = pra.linear_2D_array(mic0_center, 5, 15, 0.075)

k = 0
mic1_center = np.array([distances_from_walls[k], 3*distances_from_walls[k]])
mic1 = pra.linear_2D_array(mic1_center, 5, 30, 0.075)

k = 1
mic2_center = np.array(
    [distances_from_walls[k], 3.5*distances_from_walls[k]])
mic2 = pra.linear_2D_array(mic2_center, 5, 45, 0.075)

k = 2
mic3_center = np.array(
    [room_size[0]-1.5*distances_from_walls[k], room_size[1]-distances_from_walls[k]])
mic3 = pra.linear_2D_array(mic3_center, 5, 60, 0.075)

k = 3
mic4_center = np.array(
    [1.3*distances_from_walls[k], distances_from_walls[k]])
mic4 = pra.linear_2D_array(mic4_center, 5, 75, 0.075)

k = 4
mic5_center = np.array(
    [room_size[0]-distances_from_walls[k], 1.2*distances_from_walls[k]])
mic5 = pra.linear_2D_array(mic5_center, 5, -15, 0.075)



mics = np.zeros([2, 30])
mics[:,  0:5]  = mic0
mics[:,  5:10] = mic1
mics[:, 10:15] = mic2
mics[:, 15:20] = mic3
mics[:, 20:25] = mic4
mics[:, 25:30] = mic5
room.add_microphone_array(pra.MicrophoneArray(mics, room.fs))

print(mics)

src1 = [4, 1]
src2 = [2, 3]
src3 = [2, 4.5]

src4 = [5, 1.5]
src5 = [1.5, 1]
src6 = [5, 4]
src7 = [2, 5]

room.add_source(src1)
room.add_source(src2)
room.add_source(src3)

room.plot()

room = pra.ShoeBox(room_size)
room.add_microphone_array(pra.MicrophoneArray(mics, room.fs))
room.add_source(src4)
room.add_source(src5)
room.add_source(src6)
room.add_source(src7)


srcs = np.zeros([2, 7])
srcs[:, 0] = np.array(src1)
srcs[:, 1] = np.array(src2)
srcs[:, 2] = np.array(src3)
srcs[:, 3] = np.array(src4)
srcs[:, 4] = np.array(src5)
srcs[:, 5] = np.array(src6)
srcs[:, 6] = np.array(src7)

room.plot()
plt.show()

print(srcs)



# distances_omni = np.zeros([30, 3])
# distances_dirc = np.zeros([30, 4])
# distances = np.zeros([30, 7])
# for i in range(30):
#     for j in range(7):
#         if j < 3:
#             distances_omni[i, j] = np.linalg.norm(srcs[:, j] - mics[:, i])
#         else:
#             distances_dirc[i, j-3] = np.linalg.norm(srcs[:, j] - mics[:, i])
#         distances[i, j] = np.linalg.norm(srcs[:, j] - mics[:, i])

# plt.figure(figsize=(16, 9))
# plt.subplot(131)
# plt.hist(distances.flatten(), 20, label='all', alpha=0.8)
# plt.legend()
# plt.subplot(132)
# plt.hist(distances_omni.flatten(), 20, label='ominidirectional', alpha=0.8)
# plt.legend()
# plt.subplot(133)
# plt.hist(distances_dirc.flatten(), 20, label='directional', alpha=0.8)
# plt.legend()
# plt.show()


I = 6
J = 7

plt.plot([0, 0], [room_size[0], 0], 'k')
plt.plot([room_size[0], 0], [room_size[0], room_size[1]], 'k')
plt.plot([room_size[0], room_size[1]], [0, room_size[1]], 'k')
plt.plot([0, room_size[1]], [0, 0], 'k')

for i in range(I):
    bar = np.mean(mics[:, 5*i:5*(i+1)], axis=1)
    plt.scatter(bar[0], bar[1])
    plt.text(bar[0], bar[1], 'mic [%1.3f, %1.3f]' % (bar[0], bar[1]))

for j in range(J):
    bar = srcs[:, j]
    plt.scatter(bar[0], bar[1])
    if j < 3:
        plt.text(bar[0], bar[1], 'omn [%1.3f, %1.3f]' % (bar[0], bar[1]))
    else:
        plt.text(bar[0], bar[1], 'dir [%1.3f, %1.3f]' % (bar[0], bar[1]))

plt.show()

