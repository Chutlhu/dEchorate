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

    data_file = h5py.File(path_to_hdf5_dataset, 'r')

    # get constants and values
    room_codes = constants['datasets']
    Fs = constants['Fs']
    src_ids = constants['src_ids']
    mic_ids = constants['mic_ids']
    signals = ['rir']

    #
