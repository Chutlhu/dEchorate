from genericpath import exists
import h5py
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt

from zipfile import ZipFile
from pathlib import Path
from tqdm import tqdm

from dechorate import constants # these are stored in the __init__py file
from dechorate.stimulus import ProbeSignal
from dechorate.utils.dsp_utils import *


def get_zipped_file(filename, path_to_zipfile, path_to_output):
    
    if '020002' in str(filename):
        filename = str(filename).replace('020002', '010001f')
    
    path_to_extract = path_to_output / Path(filename)   
    if path_to_extract.exists():
        return path_to_extract
    
    with ZipFile(path_to_zipfile, 'r') as zip:
        return zip.extract(str(filename), path_to_output)


def get_wavefile_from_database(df, signals, src_ids, room_codes):

    for room_code in room_codes:
        for signal in signals:
            for src_id in src_ids:

                if room_code == '020002':
                    curr_room_code = '010001'
                    curr_fornitures = True
                else:
                    curr_room_code = room_code
                    curr_fornitures = False

                # select the annechoic recordings and only chirps
                wavefiles_db = df.loc[
                      (df['room_rfl_floor']   == int(curr_room_code[0]))
                    & (df['room_rfl_ceiling'] == int(curr_room_code[1]))
                    & (df['room_rfl_west']    == int(curr_room_code[2]))
                    & (df['room_rfl_south']   == int(curr_room_code[3]))
                    & (df['room_rfl_east']    == int(curr_room_code[4]))
                    & (df['room_rfl_north']   == int(curr_room_code[5]))
                    & (df['room_fornitures']  == curr_fornitures)
                    & (df['src_id'] == src_id)
                    & (df['src_signal'] == signal)
                ]
                
                assert len(wavefiles_db) == 31
                assert len(pd.unique(wavefiles_db['filename'])) == 1

                src_type = pd.unique(wavefiles_db['src_type'])
                assert len(src_type) == 1

                # check that there is only one filename
                if len(wavefiles_db) == 0:
                    print(wavefiles_db)
                    raise ValueError('Empty Dataframe')

                if len(wavefiles_db) > 31:
                    print(wavefiles_db)
                    raise ValueError('Multiple entries selected in the Dataframe')

                # check that there are not detrimental artifacts
                artifact = wavefiles_db['rec_artifacts'].values[0]
                if artifact > 0:
                    print('**Warnings** artifacts of type', artifact)
                
                # get only the filename                    
                filename = str(wavefiles_db['filename'].values[0])
                yield (filename, signal, src_id, room_code, artifact)


def wave_loader(wavefile, session_filename, path_to_session_zip_dir, path_to_output):

    print('Processing', wavefile, 'in', session_filename)
    # extract the file from the zip and save it
    filename = Path(session_filename, wavefile + '.wav')
    path_to_current_wavefile = get_zipped_file(filename, path_to_session_zip_dir, path_to_output)
    wav, fs = sf.read(path_to_current_wavefile)
    assert fs == constants['Fs']
    return wav, fs

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", help="Path to output files", type=str)
    parser.add_argument("--signal", help="Type of signal you wish to extract", type=str)
    parser.add_argument("--fs", help="Output sampling frequency", type=int)
    parser.add_argument("--datadir", help="Path to dEchorate data folder", type=str)
    parser.add_argument("--dbpath", help="Path to dEchorate annotation database", type=str)
    parser.add_argument("--comp", help="Compression option for hdf5", type=int, default=4)
    args = parser.parse_args()

    signal = args.signal
    new_fs = args.fs
    if not signal in constants['signals']:
        raise ValueError('Signals mus be either chirp, silence, babble, speech, noise')

    output_dir = Path(args.outdir)
    assert output_dir.exists()
    
    datadir = args.datadir
    path_to_recordings = Path(datadir)
    assert path_to_recordings.exists()

    path_to_annotation = Path(args.dbpath)
    assert path_to_recordings.exists()

    # setup paths
    path_to_tmp = Path('.', '.cache')
    if not path_to_tmp.exists():
        path_to_tmp.mkdir(parents=True, exist_ok=True)
    
    if not path_to_recordings.exists():
        raise ValueError(f'WrongPathError {path_to_recordings.resolve()} does not exist' )

    # current procesing
    curr_dset_name = f'dEchorate_{signal}'
    print('Processing:', curr_dset_name)

    # get constants and values
    room_codes = constants['datasets']
    Fs = constants['Fs']
    src_ids = constants['src_ids']
    mic_ids = constants['mic_ids']
    signals = [signal]

    if signal == 'silence':
        src_ids = constants['silence_ids']
    if signal == 'babble':
        src_ids = constants['nse_ids']

    ## Open the database
    df_note = pd.read_csv(path_to_annotation)

    ## INITIALIZE THE HDF5 DATASET
    # signal = ['chirp', 'silence', 'babble', 'speech', 'noise']

    # DATASET structure: room x mics x srcs + bonus
    # room = [000000, 010000, ..., 111111, 020002]
    # mics = [0, ..., 29, 30] : 0-29 capsule, 30th loopback
    # srcs = [0, ..., 8] : 6 dir,  3 omni
    # bonus: book for polarity

    path_to_output_dataset_hdf5 = output_dir / Path(f'{curr_dset_name}_{int(new_fs/1000)}k.h5')
    if not path_to_output_dataset_hdf5.exists():
        f = h5py.File(path_to_output_dataset_hdf5, "w")
        f.close()
    # re-open it in append mode
    hdf = h5py.File(path_to_output_dataset_hdf5, 'a')

    ## POPULATE THE HDF5 DATASET
    for (wavename, signal, src_id, room_code, artifact) in get_wavefile_from_database(df_note, signals, src_ids, room_codes):
        
        print(f'{wavename}:\tRoom: {room_code}\tSignal: {signal}\tSrc: {src_id}')
        
        group = f'/{signal}/{room_code}/{src_id:d}'

        if group in hdf:
            continue

        session_filename = f'room-{room_code}'
        path_to_session_zip = path_to_recordings / Path(session_filename+'.zip')
        wav, fs = wave_loader(wavename, session_filename, path_to_session_zip, path_to_tmp)
        
        if not new_fs == fs:
            wav = resample(wav, fs, new_fs)
        
        # here we use the default compression, so it will be faster to make operation
        hdf.attrs['signal'] = signal
        hdf.attrs['sampling_rate'] = new_fs
        hdf.create_dataset(group, data=wav, compression="gzip", compression_opts=args.comp)
        # hdf.create_dataset(group, data=wav, compression="gzip", compression_opts=4)
    
    hdf.close()
    
    print('Dataset extracted in', output_dir)