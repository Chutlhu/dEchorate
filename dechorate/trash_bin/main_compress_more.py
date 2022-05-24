from ast import comprehension
from bz2 import compress
import h5py
import argparse
from pathlib import Path

# actually you can use
# h5repack -v -f GZIP=7 dEchorate_babble.hdf5 dEchorate_babble_gzip7.hdf5

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Type of signal you wish to extract", type=str)
    args = parser.parse_args()

    print(args.file)
    path_to_old_file = Path(args.file)
    print(path_to_old_file)
    dset_in = h5py.File(path_to_old_file, mode='r')

    path_to_new_compressed_file = Path(path_to_old_file.stem + '_c9' + path_to_old_file.suffix)
    print(path_to_new_compressed_file)
    dset_out = h5py.File(path_to_new_compressed_file, 'w')

    signals = list(dset_in)

    for signal in signals:

        rooms = list(dset_in[signal])
        
        for room in rooms:

            sources = list(dset_in[signal][room])
            
            for src in sources:

                data = dset_in[signal][room][src]

                group = f'/{signal}/{room}/{src}'
                print(room, src, data.shape)
                dset_out.create_dataset(group, data=data, compression="gzip", compression_opts=9)

    dset_in.close()
    dset_out.close()