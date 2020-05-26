import numpy as np
import scipy as sp
import peakutils as pk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


from scipy.spatial import distance

from dechorate import constants
from dechorate.dataset import DechorateDataset, SyntheticDataset
from dechorate.utils.mds_utils import trilateration
from dechorate.utils.file_utils import save_to_pickle, load_from_pickle, save_to_matlab
from dechorate.utils.dsp_utils import normalize, envelope
from dechorate.utils.geo_utils import plane_from_points, mesh_from_plane, square_within_plane, dist_point_plane


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

# which dataset?
dataset_id = '011111'
L = 19556
c = constants['speed_of_sound']
Fs = constants['Fs']
recording_offset = constants['recording_offset']

# which source?
srcs_idxs = [3]
J = len(srcs_idxs)

# which microphonese?
# mics_idxs = [0, 1, 5, 6, 10, 11, 15, 16, 20, 21, 25]
mics_idxs0 = [0, 5, 10, 15, 20, 25]
mics_idxs1 = [1, 6, 11, 16, 21, 26]
mics_idxs2 = [2, 7, 12, 17, 22, 27]
mics_idxs3 = [3, 8, 13, 18, 23, 28]
mics_idxs4 = [4, 9, 14, 19, 24, 29]
mics_idxs = mics_idxs0 + mics_idxs1
# mics_idxs = range(30)

I = len(mics_idxs)
K = 7


dataset_dir = './data/dECHORATE/'
path_to_processed = './data/processed/'
path_to_note_csv = dataset_dir + 'annotations/dECHORATE_database.csv'
path_to_after_calibration = path_to_processed + 'post2_calibration/calib_output_mics_srcs_pos.pkl'

note_dict = load_from_pickle(path_to_after_calibration)
dset = DechorateDataset(path_to_processed, path_to_note_csv)
sdset = SyntheticDataset()

rirs = np.zeros([L, I, J])
mics = np.zeros([3, I])
srcs = np.zeros([3, J])
toas = np.zeros([K, I, J])

for i, m in enumerate(mics_idxs):
    for j, s in enumerate(srcs_idxs):

        # positions from beamcon
        dset.set_dataset(dataset_id)
        dset.set_entry(m, s)
        mic, src = dset.get_mic_and_src_pos()
        mics[:, i] = mic
        srcs[:, j] = src
        _, rir = dset.get_rir()

        # measure after calibration
        mics[:, i] = note_dict['mics'][:, m]
        srcs[:, j] = note_dict['srcs'][:, s]
        rirs[:, i, j] = normalize(rir)

        # print(i, 'cali', mics[:, i] - np.array(constants['room_size']))

        # double check with synthetic data
        sdset = SyntheticDataset()
        sdset.set_room_size(constants['room_size'])
        sdset.set_dataset(dataset_id, absb=1, refl=0)
        sdset.set_c(c)
        sdset.set_k_order(1)
        sdset.set_k_reflc(7)
        sdset.set_mic(mics[0, i], mics[1, i], mics[2, i])
        sdset.set_src(srcs[0, j], srcs[1, j], srcs[2, j])
        amp, tau, wall, order, gen = sdset.get_note()

        toas[:7, i, j] = tau[:7]
        toas[:7, i, j] = note_dict['toa_sym'][:7, m, s]
        toas[:7, i, j] = note_dict['toa_pck'][:7, m, s]


## COMPUTE IMAGES POSITIONS
walls = constants['refl_order_pyroom']
imgs = np.zeros([3, K, J])
prjs = np.zeros([3, K, J])
toas_imgs = np.zeros([K, I, J])
for j in range(J):
    for k in range(K):
        d = toas[k, :, j] * c
        imgs[:, k, :], error = trilateration(mics.T, d)
        for i in range(I):
            toas_imgs[k, i, j] = np.linalg.norm(imgs[:, k, j] - mics[:, i]) / c


## PLOT POSITION OF THE IMAGES
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(mics[0, :], mics[1, :], mics[2, :], marker='o', label='microphones')
ax.scatter(srcs[0, :], srcs[1, :], srcs[2, :], marker='o', label='sources')

walls = ['direct', 'ceiling', 'floor', 'west', 'south', 'east', 'north']
walls = ['direct', 'west', 'east', 'south', 'north', 'floor', 'ceiling']

for j in range(J):

    ax.scatter(imgs[0, :, j], imgs[1, :, j], imgs[2, :, j], c='C%d' % (k+2), label='images')

    for k in range(K):
        wall = walls[k]
        # ax.scatter(imgs[0, k, j], imgs[1, k, j], imgs[2, k, j], c='C%d' % (k+2), marker='x', label='img %d %s' % (k, wall))
        ax.text(imgs[0, k, j], imgs[1, k, j], imgs[2, k, j], wall)

        if k > 0:
            plt.plot([imgs[0, 0, j], imgs[0, k, j]],
                     [imgs[1, 0, j], imgs[1, k, j]],
                     [imgs[2, 0, j], imgs[2, k, j]], alpha=1-np.clip(error, 0, 0.7))

            point = (imgs[:, k, j] + imgs[:, 0, j])/2
            normal = imgs[:, k, j] - imgs[:, 0, j]

            normal = normal/np.linalg.norm(normal)

            ax.scatter(point[0], point[1], point[2], color='r', marker='x')

            a = Arrow3D([point[0], point[0] + normal[0]],
                        [point[1], point[1] + normal[1]],
                        [point[2], point[2] + normal[2]],
                        mutation_scale=10,
                        lw=1, arrowstyle="-|>", color="r")
            ax.add_artist(a)
            prjs[:, k, j] = point

a = Arrow3D([-0.5, 1.5], [0, 0], [0, 0], mutation_scale=10, lw=1, arrowstyle="-|>", color="k", alpha=0.4)
ax.add_artist(a)
a = Arrow3D([0, 0], [-0.5, 1.5], [0, 0], mutation_scale=10, lw=1, arrowstyle="-|>", color="k", alpha=0.4)
ax.add_artist(a)
a = Arrow3D([0, 0], [0, 0], [-0.5, 1], mutation_scale=10, lw=1, arrowstyle="-|>", color="k", alpha=0.4)
ax.add_artist(a)

ax.set_xlim([-0.5, 6])
ax.set_ylim([-0.5, 6])
ax.set_zlim([-0.5, 3])

plt.legend()
plt.tight_layout()
plt.savefig('./recipes/room_geometry_estimation/estimated_image.pdf', dpi=300)
plt.show()
plt.close()

# WEST
k = 1
if prjs.shape[-1] > 2:
    normal = plane_from_points(prjs[:, k, :])
    point = np.mean(prjs[:, k, :], axis=-1)
else:
    point = prjs[:, k, 0]
    normal = prjs[:, k, 0] - imgs[:, 0, 0]
    normal = normal / np.linalg.norm(normal)
V = square_within_plane(point, normal, size=(3, 3))
V = [list(zip(V[0, :], V[1, :], V[2, :]))]


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(mics[0, :], mics[1, :], mics[2, :], marker='o', label='microphones')
ax.scatter(point[0], point[1], point[2], marker='X', label='wall intercept')
ax.add_collection3d(Poly3DCollection(V, alpha=0.5, edgecolor='k'))

for j in range(J):

    ax.scatter(imgs[0, :, j], imgs[1, :, j], imgs[2, :, j], label='images')

    for k in range(7):

        wall = walls[k]
        ax.text(imgs[0, k, j], imgs[1, k, j], imgs[2, k, j], wall)

        if k == 0:
            continue

        plt.plot([imgs[0, 0, j], imgs[0, k, j]],
                [imgs[1, 0, j], imgs[1, k, j]],
                [imgs[2, 0, j], imgs[2, k, j]], alpha=1-np.clip(error, 0, 0.7), c='C%d' % (k+2))

        point = prjs[:, k, j]
        normal = prjs[:, k, j] - imgs[:, 0, j]

        normal = normal/np.linalg.norm(normal)

        ax.scatter(prjs[0, k, :], prjs[1, k, :], prjs[2, k, :], c='C%d' %(k+2), marker='x')

        a = Arrow3D([point[0], point[0] + normal[0]],
                    [point[1], point[1] + normal[1]],
                    [point[2], point[2] + normal[2]],
                    mutation_scale=10,
                    lw=1, arrowstyle="-|>", color="r")
        ax.add_artist(a)

a = Arrow3D([-0.5, 1.5], [0, 0], [0, 0], mutation_scale=10, lw=1, arrowstyle="-|>", color="k", alpha=0.4)
ax.add_artist(a)
a = Arrow3D([0, 0], [-0.5, 1.5], [0, 0], mutation_scale=10, lw=1, arrowstyle="-|>", color="k", alpha=0.4)
ax.add_artist(a)
a = Arrow3D([0, 0], [0, 0], [-0.5, 1], mutation_scale=10, lw=1, arrowstyle="-|>", color="k", alpha=0.4)
ax.add_artist(a)

ax.set_xlim([-0.5, 6])
ax.set_ylim([-0.5, 6])
ax.set_zlim([-0.5, 3])

plt.legend()
plt.tight_layout()
plt.savefig('./recipes/room_geometry_estimation/estimated_reflector.pdf' , dpi=300)
plt.show()
plt.close()


# ## all the bounding box
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(mics[0, :], mics[1, :], mics[2, :], marker='o', label='microphones')
# ax.scatter(imgs[0, 0], imgs[1, 0], imgs[2, 0], marker='o', label='srcs init')

# for k in range(1, 7):

#     print(prjs[:, k, :])

#     ax.scatter(prjs[0, k, :], prjs[1, k, :], prjs[2, k, :], c='C%d' %
#             (k+2), marker='x', label='img %d %s' % (k, wall))

#     if k in [0, 5, 6]:
#         continue

#     if prjs.shape[-1] > 2:
#         normal = plane_from_points(prjs[:, k, :])
#         point = np.mean(prjs[:, k, :], axis=-1)
#     else:
#         point =  prjs[:, k, 0]
#         normal = prjs[:, k, 0] - imgs[:, 0, 0]
#         normal = normal / np.linalg.norm(normal)
#     # xx, yy, z = mesh_from_plane(point, normal)
#     V = square_within_plane(point, normal, size=(10, 10))
#     V = [list(zip(V[0, :], V[1, :], V[2, :]))]


#     # ax.scatter(point[0], point[1], point[2], marker='o', label='mean')
#     ax.add_collection3d(Poly3DCollection(V, alpha=0.5, facecolor='C%d' % k, edgecolor='k'))

# ax.set_xlim(-1, 6)
# ax.set_ylim(-1, 6)
# ax.set_zlim(-1, 2)
# plt.show()

## COMPUTE ERRORS
# hardcoded normals
normals = {
    'direct' : np.array([ 0, 0,  0]),
    'west' : np.array([-1,  0,  0]),
    'east' : np.array([ 1,  0,  0]),
    'south': np.array([ 0, -1,  0]),
    'north': np.array([ 0,  1,  0]),
    'floor': np.array([ 0,  0, -1]),
    'ceiling': np.array([0,  0, 1]),
}
L, W, H = constants['room_size']
points = {
    'direct': np.array([0, 0,  0]),
    'west': np.array([0,  W/2,  H/2]),
    'east': np.array([L,  W/2,  H/2]),
    'south': np.array([L/2, 0,  H/2]),
    'north': np.array([L/2, W,  H/2]),
    'floor':   np.array([L/2,  W/2, 0]),
    'ceiling': np.array([L/2,  W/2, H]),
}

if J == 1:

    for k in range(1, 7):
        for j in range(J):
            wall = walls[k]
            normal = prjs[:, k, j] - imgs[:, 0, j]
            normal = normal/np.linalg.norm(normal)


            n1 = normal
            n2 = normals[wall]
            ang = np.rad2deg(np.arccos(np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))))

            wall_point = points[wall]
            dst = dist_point_plane(wall_point, prjs[:, k, j], normal)

            print(wall, '\t', dst, '\t', ang)

else:
    for k in range(1, 7):

        wall = walls[k]
        normal = plane_from_points(prjs[:, k, :])
        point = np.mean(prjs[:, k, :], axis=-1)

        n1 = normal
        n2 = normals[wall]
        ang = np.rad2deg(np.arccos(np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))))

        wall_point = points[wall]
        dst = dist_point_plane(wall_point, point, normal)

        print(wall, '\t', dst, '\t', ang)
1/0

## SKYLINE WITH NEW ESTIMATED IMAGES
L, I, J = rirs.shape
rirs_skyline = np.abs(rirs).transpose([0, 2, 1]).reshape([L, I*J])
plt.imshow(rirs_skyline, extent=[0, I*J, 0, L], aspect='auto')

# plot srcs boundaries
for j in range(J):
    plt.axvline(j*I, color='C7')

for k in range(K):
    wall = walls[k]
    # plot peak annotation
    plt.scatter(np.arange(I*J)+0.5, L - toas[k, :, :].T.flatten()*Fs, c='C%d' % (k+2), marker='x', label='%s Picking' % wall)
    plt.scatter(np.arange(I*J)+0.5, L - toas_imgs[k, :, :].T.flatten()*Fs, marker='o', facecolors='none', edgecolors='C%d' % (k+2), label='%s Pyroom' % wall)

# plt.ylim([18200, L])
plt.xlim([0, I*J])
plt.legend()
plt.show()



# distances = distance.squareform(distance.pdist(mics.T))
# print(distances)

# ## PREPARING FOR DOKMANIC:
# dokdict = {
#     'D' : distances,
#     'rirs': rirs,
#     'delay' : 0,
#     'c' : dset.c,
#     'fs' : dset.Fs,
#     'repeat' : False,
#     'mics' : mics,
#     'src' : srcs,
#     'T_direct': toas[0, :, :].squeeze(),
#     'T_reflct' : toas[1:, :, :].squeeze(),
# }

# save_to_matlab('./recipes/room_geometry_estimation/data_rooge.mat', dokdict)

# # now you can just download and run Dokmanic code
