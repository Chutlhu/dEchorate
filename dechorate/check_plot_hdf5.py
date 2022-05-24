import h5py
import argparse
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm

from dechorate import constants


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--signal", help="Type of signal you wish to extract", type=str)
    args = parser.parse_args()
    
    signal = args.signal

    Fs = constants['Fs']

    path_to_hdf = Path('.', f'dEchorate_{signal}.hdf5')
    dset = h5py.File(path_to_hdf, mode='r')

    signals = list(dset.keys())
    print('Signals', signals)
    signal = signals[0]

    rooms = list(dset[signals[0]].keys())
    print('Rooms', rooms)

    sources = list(dset[signals[0]][rooms[0]].keys())
    print('Sources', sources)
    
    dset.close()

    for room in tqdm(rooms, desc='room'):


        dset = h5py.File(path_to_hdf, mode='r')
        fig, axarr = plt.subplots(1, len(sources), figsize=(30,5))
        plt.suptitle(room)

        for s, src in enumerate(tqdm(sources, desc='src')):
    

            group = f'/{signal}/{room}/{src}'
            data = np.asarray(dset[group])
            
            if len(sources) > 1:
                ax = axarr[s]
            else:
                ax = axarr
        
            time = np.arange(data.shape[0])/Fs
            ax.set_title(f'src: {src}')
            ax.plot(time, data[:,-1] - 0.5, alpha=0.5, label='loopback')
            ax.plot(time, data[:,:5])
            ax.set_ylim([-0.5, 0.5])
            ax.legend(loc='upper left')

            del data

        plt.tight_layout()
        plt.savefig(f'./figures/{signal}_{room}_{src}.png')
        plt.close()

        dset.close()