import os
import h5py
import argparse
import numpy as np

from dechorate import constants  # these are stored in the __init__py file
from dechorate.utils.dsp_utils import resample

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--signal', dest="signal", help="Type of signal you wish to extract [in rir, speech, noise]", type=str)
    parser.add_argument(
        '--Fs', dest="Fs", help="new sampling frequency", type=int)
    args = parser.parse_args()

    # setup paths
    cwd = os.getcwd()

    signal = args.signal
    newFs = args.Fs

    path_to_hdf5_dataset = os.path.join(
        cwd, 'data', 'final', 'dechorate_with_rirs.hdf5')
    path_to_output = os.path.join(
        cwd, 'data', 'final', signal + '_matrix_dset-%s.npy'
    )

    # open the hdf5 in read only
    data_file = h5py.File(path_to_hdf5_dataset, 'r')

    # get constants and values of the dataset
    room_codes = constants['datasets']
    Fs = constants['Fs']
    src_ids = constants['src_ids']
    mic_ids = constants['mic_ids']
    signals = [signal] # even more in the future? #TODO

    # initialize the numpy array
    D = len(room_codes)
    I = len(mic_ids) - 1
    J = len(src_ids)
    L = 20 * newFs

    for d in range(D):
        print('Extracting %s' % room_codes[d])
        matrix = np.zeros([L, I, J])
        print(matrix.shape)
        for j in range(J):
            for i in range(I):

                group = '/%s/%s/%d/%d' % (room_codes[d], signal, src_ids[j], mic_ids[i])
                print(group)

                try:
                    observation = data_file[group]
                except:
                    continue

                observation = resample(observation, Fs, newFs)
                Lr = min(len(observation), L)
                matrix[:Lr,i,j] = observation[:Lr].squeeze()

        np.save(path_to_output % room_codes[d], matrix)
