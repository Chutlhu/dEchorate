import os
import h5py
import argparse
import numpy as np

from dechorate import constants  # these are stored in the __init__py file

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "signal", help="Type of signal you wish to extract", type=str)
    args = parser.parse_args()

    # setup paths
    cwd = os.getcwd()

    path_to_hdf5_dataset = os.path.join(
        cwd, 'data', 'final', 'dechorate_with_rirs.hdf5')
    path_to_output = os.path.join(
        cwd, 'data', 'final', 'rir_matrix.npy'
    )

    # open the hdf5 in read only
    data_file = h5py.File(path_to_hdf5_dataset, 'r')

    # get constants and values of the dataset
    room_codes = constants['datasets']
    Fs = constants['Fs']
    src_ids = constants['src_ids']
    mic_ids = constants['mic_ids']
    signals = ['rir']

    # initialize the numpy array
    C = len(room_codes)
    I = len(mic_ids) - 1
    J = len(src_ids)
    L = 8 * Fs
    rirs = np.zeros([C, I, J, L])
    print(rirs.shape)

    for c in range(C):
        for i in range(I):
            for j in range(J):

                group = '/%s/%s/%d/%d' % (room_codes[c], 'rir', src_ids[j], mic_ids[i])

                rir = data_file[group]
                Lr = len(rir)
                rirs[c,i,j,:Lr] = rir[:Lr].squeeze()

    np.save(path_to_output, rirs)
