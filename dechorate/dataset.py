import h5py
import numpy as np
import pandas as pd
import pyroomacoustics as pra

from dechorate import constants
from dechorate.utils.acu_utils import rt60_with_sabine, rt60_from_rirs
from dechorate.utils.dsp_utils import resample
from dechorate.utils.file_utils import load_from_pickle
from dechorate.utils.geo_utils import compute_planes, compute_image, get_point

class DechorateDataset():

    def __init__(self, path_to_data, path_to_note, path_to_mic_src_echo_note):

        # open the dataset files
        if not path_to_data.split('.')[-1] == 'hdf5':
            raise ValueError('path_to_data must be the hdf5 file')

        self.path_to_data = path_to_data
        self.dset_data = h5py.File(self.path_to_data, 'r')

        if not path_to_note.split('.')[-1] == 'csv':
            raise ValueError('path_to_note must be the csv file')
        self.path_to_note = path_to_note
        self.dset_note = pd.read_csv(self.path_to_note)


        if not path_to_mic_src_echo_note.split('.')[-1] == 'pkl':
            raise ValueError('path_to_mic_src_echo_note must be the pkl file')
        self.path_to_mic_src_echo_note = path_to_mic_src_echo_note
        self.mic_src_echo_note = load_from_pickle(self.path_to_mic_src_echo_note)

        # field of the dataset
        self.dset_note_entry = None
        self.room_code = None
        self.mic_pos = None
        self.src_pos = None
        self.i = None
        self.j = None
        self.rir = None

        # dataset constants
        self.room_size = constants['room_size']
        self.c = constants['speed_of_sound']
        self.Fs = constants['Fs']

        # equivalent synthetic dataset
        self.synth_dset = SyntheticDataset()

        self.synth_dset.set_room_size(self.room_size)
        self.synth_dset.set_c(self.c)

    def set_entry(self, room_code, mic, src):
        self.i = mic
        self.j = src
        self.room_code = room_code

    def get_entry(self, src_signal):
        f, c, w, s, e, n = [int(i) for i in self.room_code]
        entry = self.dset_note.loc[
              (self.dset_note['room_rfl_floor'] == f)
            & (self.dset_note['room_rfl_ceiling'] == c)
            & (self.dset_note['room_rfl_west'] == w)
            & (self.dset_note['room_rfl_east'] == e)
            & (self.dset_note['room_rfl_north'] == n)
            & (self.dset_note['room_rfl_south'] == s)
            & (self.dset_note['src_id'] == self.j+1)
            & (self.dset_note['mic_id'] == self.i+1)
            & (self.dset_note['src_signal'] == src_signal)
        ]
        assert len(entry) > 0
        return entry

    def get_rir(self, Fs_new =None):
        group = '%s/%s/%d/%d' % (self.room_code, 'rir', self.j+1, self.i+1)
        rir = self.dset_data[group]
        rir = rir[constants['recording_offset']:]
        if not Fs_new  is None and Fs_new  != self.Fs:
            print('Resampling with Librosa %d --> %d' % (self.Fs, Fs_new))
            rir = resample(rir, self.Fs, Fs_new)
        self.rir = rir.squeeze()
        return rir

    def get_mic_and_src_pos(self, updated=True):
        self.mic_pos = self.mic_src_echo_note['mics'][:, self.i]
        self.src_pos = self.mic_src_echo_note['srcs'][:, self.j]
        return self.mic_pos, self.src_pos

    def get_echo(self, kind='pck', order=0):
        if not kind in ['sym', 'pck']:
            raise ValueError('Kind must be either sym or pck')
        if kind == 'sym':
            # toas = self.mic_src_echo_note['toa_sym'][:, self.i, self.j]
            toas, amps, walls = self.get_synth_note()
        if kind == 'pck':
            toas = self.mic_src_echo_note['toa_pck'][:, self.i, self.j]
        if order > 0:
            raise NotImplementedError
        return toas


    def get_synth_echo(self, walls):
        m = self.mic_pos.copy()
        s = self.src_pos.copy()


        # W = len(walls)
        # toas = np.zeros(W)
        # for w, wall in enumerate(walls):
        #     if not wall in constants['refl_order_calibr']:
        #         raise ValueError('Wall must be either  "c", "f", "w", "s", "e", or "n"')

        #     if wall == 'd':
        #         im = get_point(m)
        #     else:
        #         plane = compute_planes(constants['room_size'])[wall]
        #         im = compute_image(m, plane)

        #     toas[w] = float(im.distance(s).evalf())/constants['speed_of_sound']

        return toas

    def get_synth_note(self):
        sdset = SyntheticDataset()
        sdset.set_room_size(self.room_size)
        sdset.set_dataset(self.room_code, absb=0.9, refl=0.1)
        sdset.set_c(self.c)
        sdset.set_k_order(1)
        sdset.set_k_reflc(7)
        sdset.set_mic(self.mic_pos[0], self.mic_pos[1], self.mic_pos[2])
        sdset.set_src(self.src_pos[0], self.src_pos[1], self.src_pos[2])
        taus, amps, walls = sdset.get_note(False, 'pra_order')
        return taus, amps

    def compute_rt60(self, M=100, snr=45, do_schroeder=True, val_min=-90):
        if self.rir is None:
            raise ValueError('RIR not retrieved yet. call get_rir\(\) explicitly')
        return rt60_from_rirs(self.rir, self.Fs, M=M, snr=snr, do_schroeder=do_schroeder, val_min=val_min)


class SyntheticDataset():
    def __init__(self):
        self.x = [2, 2, 2]
        self.s = [4, 3, 1]
        self.Fs = 48000
        self.c = 343
        self.k_order = None
        self.k_reflc = None

        self.room_size = [5, 4, 6]
        self.amp = None
        self.toa = None
        self.order = None

        self.rir = None

        self.absorption = {
            'north': 0.8,
            'south': 0.8,
            'east': 0.8,
            'west': 0.8,
            'floor': 0.8,
            'ceiling': 0.8,
        }

    def set_abs(self, wall, abs_coeff):
        self.absorption[wall] = abs_coeff

    def set_room_size(self, room_size):
        self.room_size = room_size

    def set_fs(self, fs):
        self.Fs = fs

    def set_c(self, c):
        self.c = c

    def set_mic(self, x, y, z):
        self.x[0] = x
        self.x[1] = y
        self.x[2] = z

    def set_src(self, x, y, z):
        self.s[0] = x
        self.s[1] = y
        self.s[2] = z

    def set_k_reflc(self, K):
        self.k_reflc = K

    def set_k_order(self, K):
        self.k_order = K

    def set_dataset(self, dset_code, absb=0.2, refl=0.8):
        f, c, w, s, e, n = [int(i) for i in list(dset_code)]
        self.absorption = {
            'north': refl if n else absb,
            'south': refl if s else absb,
            'east': refl if e else absb,
            'west': refl if w else absb,
            'floor': refl if f else absb,
            'ceiling': refl if c else absb,
        }

    def make_room(self):
        room = pra.ShoeBox(
            self.room_size, fs=self.Fs,
            absorption=self.absorption, max_order=self.k_order)
        room.set_sound_speed(self.c)
        room.add_microphone_array(
            pra.MicrophoneArray(
                np.array(self.x)[:, None], room.fs))

        room.add_source(self.s)

        return room

    def plot_room(self, ax3D):
        ax3D.set_xlim([-0.1, self.room_size[0]+0.1])
        ax3D.set_ylim([-0.1, self.room_size[1]+0.1])
        ax3D.set_zlim([-0.1, self.room_size[2]+0.1])

        ax3D.scatter(self.x[0], self.x[1], self.x[2], marker='o', label='mics')
        ax3D.scatter(self.s[0], self.s[1], self.s[2], marker='x', label='srcs')

        # plot lines of the room
        ax3D.plot([0, self.room_size[0]], [0, 0], zs=[
                  0, 0], color='black', alpha=0.6)
        ax3D.plot([0, 0], [0, self.room_size[1]], zs=[
                  0, 0], color='black', alpha=0.6)
        ax3D.plot([0, 0], [0, 0], zs=[0, self.room_size[2]],
                  color='black', alpha=0.6)

        # plot line of the image source
        # direct path
        ax3D.plot([self.x[0], self.s[0]], [self.x[1], self.s[1]], zs=[
                  self.x[2], self.s[2]], c='C1', ls='--', alpha=0.8)

        return

    def get_rir(self, normalize : bool):
        room = self.make_room()
        room.image_source_model()
        room.compute_rir()
        rir = room.rir[0][0]
        rir = rir[40:]
        self.rir = rir
        if normalize:
            return np.arange(len(rir))/self.Fs, rir/np.max(np.abs(rir))
        else:
            return np.arange(len(rir))/self.Fs, rir

    def get_walls_name_from_id(self, wall_id : int):
        if wall_id == -1:
            return 'direct'
        return self.wallsId[wall_id]
        # print(wall_id)
        # for wall_name in self.wallsId:
        #     curr_wall_id = self.wallsId[wall_name]
        #     if int(curr_wall_id) == int(wall_id):
        #         return wall_name


    def get_wall_order_from_images(self, images, order, room_size):

        print(order)
        print(room_size)
        planes = compute_planes(room_size)
        img0 = images[:, np.where(order == 0)].squeeze()
        D, K = images.shape
        walls = []
        for k in range(K):
            bar = (img0 + images[:, k])/2
            dist = np.linalg.norm(img0 - images[:, k])
            if dist < 1e-16:
                walls.append('d')
                continue

            bar = get_point(bar)
            for p in planes:
                P = planes[p]

                if P is None:
                    continue
                else:
                    dist = float(P.distance(bar).evalf())
                    if dist == 0:
                        walls.append(p)
        assert len(walls) == K
        return walls

    def get_note(self, ak_normalize: bool = False, tk_order : str = 'pra_order'):

        room = self.make_room()
        room.image_source_model()
        self.wallsId = room.wall_names

        assert room.mic_array.R.shape[1] == 1
        assert len(room.sources) == 1

        j = 0
        K = self.k_reflc
        source = room.sources[j]

        images = source.images
        orders = source.orders
        dampings = source.damping

        pra_order = constants['refl_order_pyroom']
        walls = self.get_wall_order_from_images(images,orders,self.room_size)
        


        distances = np.linalg.norm(
            images - room.mic_array.R, axis=0)


        tk = distances / self.c
        dk = dampings.squeeze()
        ak = dk / (distances)

        # walls = []
        # order = np.zeros(K)
        # generators = []

        # order for location
        if tk_order == 'earliest':
            indices = np.argsort(tk)
        elif tk_order == 'pra_order':
            indices = np.argsort(orders)
        elif tk_order == 'strongest':
            indices = np.argsort(np.abs(ak))[::-1]
        else:
            raise ValueError('Wrong ordering option')

        tk = tk[indices[:K]]
        dk = dk[indices[:K]]
        ak = ak[indices[:K]]

        # for o, k in enumerate(ordering):
            # print(room.sources[j].damping)
            # amp[o] = room.sources[j].damping[k] / (4 * np.pi * distances[k])
            # toa[o] = distances[k]/self.c
            # order[o] = room.sources[j].orders[k]

            # wall_id = self.get_walls_name_from_id(room.sources[j].walls[k])
            # wall_sequence = room.sources[j].wall_sequence(k)
            # wall_sequence = [self.get_walls_name_from_id(wall) for wall in wall_sequence]
            # wall_sequence = wall_sequence
            # if len(wall_sequence) > 1:
            #     wall_sequence = wall_sequence[:-1]
            # wall_sequence = '_'.join(wall_sequence)
            # generators.append(wall_sequence)
            # walls.append(wall_id)

            # assert wall_id == generators[o][0]
            # assert 'direct' == generators[o][-1]


        if ak_normalize:
            ak = ak/np.max(np.abs(ak))

        return tk, ak

    def get_rt60_sabine(self):
        if self.rir is None:
            raise ValueError('RIR not retrieved yet. call get_rir\(\) explicitly')
        return rt60_with_sabine(self.room_size, self.absorption)


    def compute_rt60(self, M=100, snr=45, do_schroeder=True, val_min=-90):
        if self.rir is None:
            raise ValueError(
                'RIR not retrieved yet. call get_rir\(\) explicitly')

        return rt60_from_rirs(self.rir, self.Fs, M=M, snr=snr, do_schroeder=do_schroeder, val_min=val_min)

if __name__ == "__main__":
    dset = SyntheticDataset()
    dset.set_k_order(2)
    dset.set_k_reflc(100)

    amp, toa, wall, order, generators = dset.get_note()

    pass
