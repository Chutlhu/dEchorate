import h5py
import numpy as np
import pandas as pd
import pyroomacoustics as pra

from dechorate import constants
from dechorate.utils.acu_utils import rt60_with_sabine, rt60_from_rirs
from dechorate.utils.dsp_utils import resample

class DechorateDataset():

    def __init__(self, path_to_processed, path_to_note_csv):
        self.Fs = 48000
        self.path_to_processed = path_to_processed
        self.path_to_note_csv = path_to_note_csv
        self.dataset_data = None
        self.dataset_note = None
        self.entry = None
        self.mic_pos = None
        self.src_pos = None
        self.mic_i = 0
        self.src_j = 0
        self.rir = None


        self.room_size = constants['room_size']
        self.c = constants['speed_of_sound']
        self.Fs = constants['Fs']

        self.sdset = SyntheticDataset()
        self.sdset.set_room_size(self.room_size)
        self.sdset.set_c(self.c)

    def set_dataset(self, dset_code):
        path_to_data_hdf5 = self.path_to_processed + '%s_rir_data.hdf5' % dset_code
        dset_rir = h5py.File(path_to_data_hdf5, 'r')
        dset_note = pd.read_csv(self.path_to_note_csv)
        f, c, w, s, e, n = [int(i) for i in list(dset_code)]
        dset_note = dset_note.loc[
            (dset_note['room_rfl_floor'] == f)
            & (dset_note['room_rfl_ceiling'] == c)
            & (dset_note['room_rfl_west'] == w)
            & (dset_note['room_rfl_east'] == e)
            & (dset_note['room_rfl_north'] == n)
            & (dset_note['room_rfl_south'] == s)
            & (dset_note['room_fornitures'] == False)
            & (dset_note['src_signal'] == 'chirp')
        ]
        assert len(dset_note) > 0
        self.dataset_data = dset_rir
        self.dataset_note = dset_note
        self.dset_code = dset_code
        self.sdset.set_dataset(dset_code)

    def set_entry(self, i, j):
        self.mic_i = i
        self.src_j = j
        self.entry = self.dataset_note.loc[(
            self.dataset_note['src_id'] == j+1) & (self.dataset_note['mic_id'] == i+1)]

    def get_rir(self, Fs=None):
        wavefile = self.entry['filename'].values[0]
        rir = self.dataset_data['rir/%s/%d' % (wavefile, self.mic_i)][()].squeeze()
        rir = rir[constants['recording_offset']:]
        if not Fs is None and Fs != self.Fs:
            print('Resampling %d --> %d' % (self.Fs, Fs))
            rir = resample(rir, self.Fs, Fs)
        self.rir = rir
        return rir

    def get_file_database_id(self, src_signal):
        dset_note = pd.read_csv(self.path_to_note_csv)
        f, c, w, s, e, n = [int(i) for i in list(self.dset_code)]
        dset_note = dset_note.loc[
            (dset_note['room_rfl_floor'] == f)
            & (dset_note['room_rfl_ceiling'] == c)
            & (dset_note['room_rfl_west'] == w)
            & (dset_note['room_rfl_east'] == e)
            & (dset_note['room_rfl_north'] == n)
            & (dset_note['room_rfl_south'] == s)
            & (dset_note['room_fornitures'] == False)
            & (dset_note['src_id'] == self.src_j + 1)
            & (dset_note['mic_id'] == self.mic_i + 1)
            & (dset_note['src_signal'] == src_signal)
        ]
        for filename in dset_note['filename']:
            yield filename

    def get_diffuse_noise(self, duration=20):
        dset_note = pd.read_csv(self.path_to_note_csv)
        f, c, w, s, e, n = [int(i) for i in list(self.dset_code)]
        dset_note = dset_note.loc[
            (dset_note['room_rfl_floor'] == f)
            & (dset_note['room_rfl_ceiling'] == c)
            & (dset_note['room_rfl_west'] == w)
            & (dset_note['room_rfl_east'] == e)
            & (dset_note['room_rfl_north'] == n)
            & (dset_note['room_rfl_south'] == s)
            & (dset_note['room_fornitures'] == False)
            & (dset_note['src_id'] == self.src_j + 1)
            & (dset_note['mic_id'] == self.mic_i + 1)
            & (dset_note['src_signal'] == 'noise')
        ]
        noise_filename = dset_note['filename'].unique()
        print(noise_filename)
        1/0

    def get_mic_and_src_pos(self):
        self.mic_pos = np.array([self.entry['mic_pos_x'].values + constants['offset_beacon'][0],
                                 self.entry['mic_pos_y'].values + constants['offset_beacon'][1],
                                 self.entry['mic_pos_z'].values + constants['offset_beacon'][2]]).squeeze()
        self.src_pos = np.array([self.entry['src_pos_x'].values + constants['offset_beacon'][0],
                                 self.entry['src_pos_y'].values + constants['offset_beacon'][1],
                                 self.entry['src_pos_z'].values + constants['offset_beacon'][2]]).squeeze()


        # self.mic_pos = np.array([self.entry['mic_pos_x'].values,
        #                          self.entry['mic_pos_y'].values,
        #                          self.entry['mic_pos_z'].values]).squeeze()
        # self.src_pos = np.array([self.entry['src_pos_x'].values,
        #                          self.entry['src_pos_y'].values,
        #                          self.entry['src_pos_z'].values]).squeeze()

        self.sdset.set_mic(self.mic_pos[0], self.mic_pos[1], self.mic_pos[2])
        self.sdset.set_src(self.src_pos[0], self.src_pos[1], self.src_pos[2])
        return self.mic_pos, self.src_pos

    def get_ism_annotation(self, k_order=1, k_reflc=7):
        self.sdset.set_k_order(k_order)
        self.sdset.set_k_reflc(k_reflc)
        amp, toa, walls, order, generators = self.sdset.get_note()
        return amp, toa, walls, order,generators

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
