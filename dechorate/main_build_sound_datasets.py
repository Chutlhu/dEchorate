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


def get_wavefile_from_database(df, signals, mic_ids, src_ids, room_codes):

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
    return wav


# def build_rir_hdf5(wavefile_chirps, path_to_output_anechoic_dataset, path_to_output_anechoic_dataset_rir):
#     f_raw = h5py.File(path_to_output_anechoic_dataset, 'r')
#     f_rir = h5py.File(path_to_output_anechoic_dataset_rir, 'w')
#     f_rir = h5py.File(path_to_output_anechoic_dataset_rir, 'a')

#     # recreate the probe signal
#     Fs = 48000
#     ps = ProbeSignal('exp_sine_sweep', Fs)
#     n_seconds = 10
#     amplitude = 0.7
#     n_repetitions = 3
#     silence_at_start = 2
#     silence_at_end = 2
#     sweeprange = [100, 14e3]
#     times, signal = ps.generate(n_seconds, amplitude, n_repetitions,
#                                 silence_at_start, silence_at_end, sweeprange)

#     for wavefile in wavefile_chirps:
#         x = f_raw['recordings/48k/' + wavefile]

#         # compute the global delay from the playback:
#         playback = x[:, -1]
#         rir_playback = ps.compute_rir(playback[:, None])
#         delay_sample = np.argmax(np.abs(rir_playback))
#         # delay_sample = ps.compute_delay(playback[:, None], start=1, duration=10)
#         delay = delay_sample/Fs
#         print(delay_sample)

#         try:
#             assert delay > 0
#         except:
#             plt.plot(normalize(playback))
#             plt.plot(normalize(ps.signal))
#             plt.show()

#         # estimate the rir in the remaining channels:
#         for i in tqdm(range(30)):
#             recording = x[:, i]

#             rir_i = ps.compute_rir(recording[:, None])
#             # for anechoic signal crop after 1 second
#             rir_i = rir_i[0:int(0.5*Fs)]

#             # store info in the anechoic dataset
#             f_rir.create_dataset('rir/%s/%d' % (wavefile, i), data=rir_i)
#             f_rir.create_dataset('delay/%s/%d' % (wavefile, i), data=delay_sample)
#             plt.show()
#     return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("signal", help="Type of signal you wish to extract", type=str)
    args = parser.parse_args()

    signal = args.signal
    if not signal in constants['signals']:
        raise ValueError('Signals mus be either chirp, silence, babble, speech, noise')

    # setup paths
    path_to_tmp = Path('.', '.cache')
    path_to_output = Path('.')
    path_to_recordings = Path('/','home','chutlhu','Documents','Datasets','dEchorate','raw')
    path_to_annotation = Path('.', 'data','dEchorate_database.csv')
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
        src_ids = [99]
    if signal == 'babble':
        src_ids = [1,2,3,4]

    ## Open the database
    df_note = pd.read_csv(path_to_annotation)

    ## INITIALIZE THE HDF5 DATASET
    # signal = ['chirp', 'silence', 'babble', 'speech', 'noise']

    # DATASET structure: room x mics x srcs + bonus
    # room = [000000, 010000, ..., 111111, 020002]
    # mics = [0, ..., 29, 30] : 0-29 capsule, 30th loopback
    # srcs = [0, ..., 8] : 6 dir,  3 omni
    # bonus: book for polarity

    path_to_output_dataset_hdf5 = path_to_output / Path(f'{curr_dset_name}.hdf5')
    if not path_to_output_dataset_hdf5.exists():
        f = h5py.File(path_to_output_dataset_hdf5, "w")
        f.close()
    # re-open it in append mode
    hdf = h5py.File(path_to_output_dataset_hdf5, 'a')

    ## POPULATE THE HDF5 DATASET
    for (wavename, signal, src_id, room_code, artifact) in get_wavefile_from_database(df_note, signals, mic_ids, src_ids, room_codes):
        
        print(f'{wavename}:\tRoom: {room_code}\tSignal: {signal}\tSrc: {src_id}')
        
        group = f'/{room_code}/{signal}/{src_id:d}'

        if group in hdf:
            continue

        session_filename = f'room-{room_code}'
        path_to_session_zip = path_to_recordings / Path(session_filename+'.zip')
        wav = wave_loader(wavename, session_filename, path_to_session_zip, path_to_tmp)
        
        hdf.create_dataset(group, data=wav, compression="gzip", compression_opts=7)


        #     print(wavename, signal, mic_id, src_id, room_code, artifact)
    
        #     print('Done with extraction')

    # if signal == 'rir':

    #     path_to_note = os.path.join(cwd, 'data', 'final', 'manual_annatotion.csv')
    #     mics_here = [1]  # the wavefile collects all the channels already

    #     df = pd.DataFrame()
    #     c = 0

    #     for (wavename, signal, mic_id, src_id, room_code, artifact) in get_wavefile_from_database(path_to_note, ['chirp'], mics_here, src_ids, room_codes):

    #         for mic_id in mic_ids:

    #             # get the chirp recording in the hdf5
    #             if mic_id == 31:
    #                 path_to_rec_in_hdf5 = '/%s/%s/%d/%s' % (room_code,signal,src_id,'loopback')
    #             else:
    #                 path_to_rec_in_hdf5 = '/%s/%s/%d/%d' % (room_code,signal,src_id,mic_id)

    #             print(path_to_rec_in_hdf5)
    #             rec = np.array(data_file[path_to_rec_in_hdf5])

    #             # get the corresponding loopback signal
    #             path_to_rec_in_hdf5 = '/%s/%s/%d/%s' % (room_code,signal,src_id,'loopback')
    #             loop = np.array(data_file[path_to_rec_in_hdf5])

    #             # RIR estimation
    #             Fs = constants['rir_processing']['Fs']
    #             assert Fs == constants['Fs']
    #             n_seconds = constants['rir_processing']['n_seconds']
    #             amplitude = constants['rir_processing']['amplitude']
    #             n_repetitions = constants['rir_processing']['n_repetitions']
    #             silence_at_start = constants['rir_processing']['silence_at_start']
    #             silence_at_end = constants['rir_processing']['silence_at_end']
    #             sweeprange = constants['rir_processing']['sweeprange']
    #             stimulus = constants['rir_processing']['stimulus']
    #             ps = ProbeSignal(stimulus, Fs)

    #             times, s = ps.generate(n_seconds, amplitude, n_repetitions, silence_at_start, silence_at_end, sweeprange)

    #             # compute the global delay from the rir with the playback signal
    #             try:
    #                 rir_loop = ps.compute_rir(loop[:,None], windowing=False)
    #                 delay = np.argmax(np.abs(rir_loop)) # in samples
    #             except:
    #                 delay = 999999

    #             # compute the rir
    #             rir = ps.compute_rir(rec[:, None], windowing=False)
    #             rir = rir[0:int(5*Fs)]

    #             if mic_id == 31:
    #                 group = '/%s/%s/%d/%s' % (room_code, 'rir', src_id, 'loopback')
    #             else:
    #                 group = '/%s/%s/%d/%d' % (room_code, 'rir', src_id, mic_id)
    #             try:
    #                 data_file.create_dataset(group, data=rir)
    #             except Exception as e:
    #                 print(e)
    #                 print(group)

    #             df.at[c, 'signal'] = 'rir'
    #             df.at[c, 'wavefile'] = wavename
    #             df.at[c, 'path_hdf5'] = group
    #             df.at[c, 'mic_id'] = mic_id
    #             df.at[c, 'src_id'] = src_id
    #             df.at[c, 'room_code'] = room_code
    #             df.at[c, 'delay'] = delay
    #             df.at[c, 'artifact'] = artifact
    #             c += 1

    #             df.to_csv(os.path.join(path_to_output, 'database_delay.csv'))


    pass
