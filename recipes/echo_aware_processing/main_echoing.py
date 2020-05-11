import numpy as np
import scipy as sp
import peakutils as pk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


from dechorate import constants
from dechorate.dataset import DechorateDataset, SyntheticDataset
from dechorate.utils.mds_utils import trilateration
from dechorate.utils.file_utils import save_to_pickle, load_from_pickle, save_to_matlab
from dechorate.utils.dsp_utils import normalize, envelope


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

# import constants
L = 19556
c = constants['speed_of_sound']
Fs = constants['Fs']
recording_offset = constants['recording_offset']

# which dataset?
dataset_id = '011111'

# which source?
srcs_idxs = [0]
J = len(srcs_idxs)

# which microphonese?
mics_idxs0 = np.arange(0, 5)
mics_idxs = mics_idxs0

I = len(mics_idxs)

# how many reflections? Lets consider 7
K = 7

## IMPORT DATA
dataset_dir = './data/dECHORATE/'
path_to_processed = './data/processed/'
path_to_note_csv = dataset_dir + 'annotations/dECHORATE_database.csv'
path_to_after_calibration = path_to_processed + \
    'post2_calibration/calib_output_mics_srcs_pos.pkl'

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
        rirs[:, i, j] = rir

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


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(mics[0, :], mics[1, :], mics[2, :], marker='o', label='mics init')
ax.scatter(srcs[0, :], srcs[1, :], srcs[2, :], marker='o', label='srcs init')

walls = constants['refl_order_pyroom']
imgs = np.zeros([3, K, J])
toas_imgs = np.zeros([K, I, J])
for j in range(J):
    for k in range(K):
        d = toas[k, :, j] * c
        imgs[:, k, :], error = trilateration(mics.T, d)
        print(error)
        wall = walls[k]
        ax.scatter(imgs[0, k, j], imgs[1, k, j], imgs[2, k, j], c='C%d' % (
            k+2), marker='x', label='img %d %s' % (k, wall))
        for i in range(I):
            toas_imgs[k, i, j] = np.linalg.norm(imgs[:, k, j] - mics[:, i]) / c

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

            # d = -np.sum(point*normal)  # dot product
            # # create x,y
            # xx, yy = np.meshgrid(range(-2, 2), range(-2, 2))

            # # calculate corresponding z
            # z1 = (-normal[0]*xx - normal[1]*yy - d)*1./normal[2]

            # ax.plot_surface(xx, yy, z1, color='blue', alpha=0.1, zorder=k)

            ax.set_xlim([-0.5, 6])
            ax.set_ylim([-0.5, 6])
            ax.set_zlim([-0.5, 3])


plt.legend()
plt.tight_layout()
plt.savefig('./recipes/room_geometry_estimation/estimated_images.pdf', dpi=300)
plt.show()

for i in range(I):
    errs = np.abs(toas_imgs[:, i, 0] - toas[:, i, 0])*c
    print(['%1.3f' % k for k in errs])
    # print(np.mean(errs))
    # print(np.max(errs))


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
    plt.scatter(np.arange(I*J)+0.5, L - toas[k, :, :].T.flatten()
                * Fs, c='C%d' % (k+2), marker='x', label='%s Picking' % wall)
    plt.scatter(np.arange(I*J)+0.5, L - toas_imgs[k, :, :].T.flatten(
    )*Fs, marker='o', facecolors='none', edgecolors='C%d' % (k+2), label='%s Pyroom' % wall)

# plt.ylim([18200, L])
plt.xlim([0, I*J])
plt.legend()
plt.show()


distances = distance.squareform(distance.pdist(mics.T))
print(distances)

## PREPARING FOR DOKMANIC:
dokdict = {
    'D': distances,
    'rirs': rirs,
    'delay': 0,
    'c': dset.c,
    'fs': dset.Fs,
    'repeat': False,
    'mics': mics,
    'src': srcs,
    'T_direct': toas[0, :, :].squeeze(),
    'T_reflct': toas[1:, :, :].squeeze(),
}

save_to_matlab('./recipes/room_geometry_estimation/data_rooge.mat', dokdict)

# now you can just download and run Dokmanic code
