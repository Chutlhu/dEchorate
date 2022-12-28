import h5py
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm

from dechorate import constants # these are stored in the __init__py file
from dechorate.stimulus import ProbeSignal
from dechorate.utils.dsp_utils import *

rec_offset = constants['recording_offset']
L = int(1.*constants['Fs'])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", help="Path to output files", type=str)
    parser.add_argument("--dbpath", help="Path to dEchorate database", type=str)
    parser.add_argument("--chirps", help="Path to dEchorate_chirp.h5", type=str)
    parser.add_argument("--comp", help="Compression option for h5", type=int, default=4)
    args = parser.parse_args()

    curr_dset_name = 'dEchorate_rirs'

    # setup paths
    path_to_output = Path(args.outdir)
    assert path_to_output.exists()
    path_to_annotation = Path(args.dbpath)
    assert path_to_annotation.exists()
    path_to_chirps = Path(args.chirps)
    assert path_to_chirps.exists()
    
    # get constants and values
    room_codes = constants['datasets']
    Fs = constants['Fs']
    src_ids = constants['src_ids']
    mic_ids = constants['mic_ids']

    # open the database
    df_note = pd.read_csv(path_to_annotation)

    # open the dataset of chirsp
    dset_chirp = h5py.File(path_to_chirps, "r")

    ## INITIALIZE THE HDF5 DATASET
    # signal = ['chirp', 'silence', 'babble', 'speech', 'noise']

    # DATASET structure: room x mics x srcs + bonus
    # room = [000000, 010000, ..., 111111, 020002]
    # mics = [0, ..., 29, 30] : 0-29 capsule, 30th loopback
    # srcs = [0, ..., 8] : 6 dir,  3 omni
    # bonus: book for polarity

    path_to_output_dataset_h5 = path_to_output / Path(f'{curr_dset_name}.h5')
    if not path_to_output_dataset_h5.exists():
        f = h5py.File(path_to_output_dataset_h5, 'w')
        f.close()
    # re-open it in append mode
    hdf = h5py.File(path_to_output_dataset_h5, 'a')

    ## POPULATE THE HDF5 DATASET

    # navigate the chirps dataset
    dt = {
        'signal' : []
    ,   'room_code' : []
    ,   'src_id' : []
    ,   'mic_id' : []
    ,   'path_hdf5' : []
    }

    for room_code in tqdm(room_codes, desc="room_code"):

        for src_id in tqdm(src_ids, desc="src_id"):

            group = f'/chirp/{room_code}/{src_id:d}'
            data = np.array(dset_chirp[group])
        
            loopback = data[:,-1]

            # RIR estimation
            Fs = constants['rir_processing']['Fs']
            assert Fs == constants['Fs']
            n_seconds = constants['rir_processing']['n_seconds']
            amplitude = constants['rir_processing']['amplitude']
            n_repetitions = constants['rir_processing']['n_repetitions']
            silence_at_start = constants['rir_processing']['silence_at_start']
            silence_at_end = constants['rir_processing']['silence_at_end']
            sweeprange = constants['rir_processing']['sweeprange']
            stimulus = constants['rir_processing']['stimulus']
            ps = ProbeSignal(stimulus, Fs)
            times, s = ps.generate(n_seconds, amplitude, n_repetitions, silence_at_start, silence_at_end, sweeprange)
            
            # compute the global delay from the rir with the playback signal
            try:
                rir_loop = ps.compute_rir(loopback[:,None], windowing=False)
                delay = np.argmax(np.abs(rir_loop)) # in samples
            except:
                # print(delay)
                delay = 999999

            n_mics = data.shape[1]
            rirs = np.zeros([5*Fs, n_mics])

            group = f'/rir/{room_code}/{src_id:d}'

            for i in tqdm(range(n_mics), desc="i"):

                if int(room_code) == 20002:
                    curr_room_code = 20002
                    curr_fornitures = True
                else:
                    curr_room_code = room_code
                    curr_fornitures = False

                info = df_note.loc[
                    (df_note['src_id'] == float(src_id))
                &   (df_note['src_signal'] == 'chirp')
                &   (df_note['room_code'] == int(curr_room_code))
                &   (df_note['room_fornitures'] == curr_fornitures)
                &   (df_note['mic_id'] == float(i))
                ]

                try:
                    assert len(info) == 1
                except:
                    print(room_code, curr_room_code, src_id, i)
                    print(info)
                    hdf.close()
                    exit()

                # # compute the rir
                if group in hdf:
                    continue
                else:
                    rir = ps.compute_rir(data[:, i, None], windowing=False)
                    rirs[:,i] = rir[0:int(5*Fs)].squeeze()


                dt['signal'].append('rir')
                dt['room_code'].append(room_code)
                dt['src_id'].append(src_id)
                dt['mic_id'].append(i)
                dt['path_hdf5'].append(group)

            # compensate delays (hardcoded in __init__.py)
            if group in rec_offset.keys():
                d = rec_offset[group]
            else:
                d = rec_offset['standard']

            rirs = rirs[d:d+L,:]
            
            if group in hdf:
                continue
            else:
                hdf.create_dataset(group, data=rirs, compression="gzip", compression_opts=args.comp)

            df = pd.DataFrame(dt)
            df.to_csv(path_to_output / Path('dEchorate_rir_database.csv'))
            

    hdf.close()