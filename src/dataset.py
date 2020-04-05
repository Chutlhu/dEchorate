import h5py
import numpy as np
import pandas as pd
import pyroomacoustics as pra


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

    def set_entry(self, i, j):
        self.mic_i = i
        self.src_j = j
        self.entry = self.dataset_note.loc[(
            self.dataset_note['src_id'] == j+1) & (self.dataset_note['mic_id'] == i+1)]

    def get_rir(self):
        wavefile = self.entry['filename'].values[0]
        self.rir = self.dataset_data['rir/%s/%d' % (wavefile, self.mic_i)][()].squeeze()
        self.rir = self.rir[4444:]
        # rir_abs = np.abs(rir[6444:])
        # self.rir = rir_abs/np.max(rir_abs)
        return np.arange(len(self.rir))/self.Fs, self.rir

    def get_mic_and_src_pos(self):
        self.mic_pos = np.array([self.entry['mic_pos_x'].values,
                                 self.entry['mic_pos_y'].values, self.entry['mic_pos_z'].values]).squeeze()

        self.src_pos = np.array([self.entry['src_pos_x'].values,
                                 self.entry['src_pos_y'].values, self.entry['src_pos_z'].values]).squeeze()
        return self.mic_pos, self.src_pos


class SyntheticDataset:
    def __init__(self):
        self.x = [2, 2, 2]
        self.s = [4, 3, 1]
        self.Fs = 48000
        self.c = 343
        self.k_order = None
        self.k_reflc = None

        self.room_size = None
        self.amp = None
        self.toa = None
        self.order = None

        self.absorption = {
            'north': 0.8,
            'south': 0.8,
            'east': 0.8,
            'west': 0.8,
            'floor': 0.8,
            'ceiling': 0.8,
        }

    def set_room_size(self, room_size):
        self.room_size = room_size

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

    def get_rir(self):
        room = self.make_room()
        room.image_source_model()
        room.compute_rir()
        rir = room.rir[0][0]
        rir = rir[40:]
        return np.arange(len(rir))/self.Fs, rir/np.max(np.abs(rir))

    def get_walls_name_from_id(self, wallsId, wall_id):
        for wall_name in wallsId:
            curr_wall_id = wallsId[wall_name]
            if int(curr_wall_id) == int(wall_id):
                return wall_name


    def get_note(self):
        room = self.make_room()
        room.image_source_model(use_libroom=False)

        j = 0
        K = self.k_reflc
        toa = np.zeros(K)
        amp = np.zeros(K)
        walls = []
        order = np.zeros(K)
        images = room.sources[j].images
        center = room.mic_array.center
        distances = np.linalg.norm(
            images - room.mic_array.R, axis=0)
        # order for location
        ordering = np.argsort(distances)[:K]
        # order for orders
        ordering = np.argsort(room.sources[j].orders)[:K]
        for o, k in enumerate(ordering):
            amp[o] = room.sources[j].damping[k] / (4 * np.pi * distances[k])
            toa[o] = distances[k]/self.c
            wall_id = self.get_walls_name_from_id(room.wallsId, room.sources[j].walls[k])
            if wall_id is None:
                wall_id = 'direct'
            walls.append(wall_id)
            order[o] = room.sources[j].orders[k]

        amp = amp/amp[0]

        return amp, toa, walls, order
