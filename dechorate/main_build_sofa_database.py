import h5py
import argparse
import platform

import sofar as sf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from tqdm import tqdm
from pathlib import Path
from dechorate import constants
from dechorate.utils.file_utils import load_from_pickle

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("toa", help="Path to dEchorate csv echo annotation", type=str)
    parser.add_argument("csv", help="Path to dEchorate csv database", type=str)
    parser.add_argument("hdf", help="Path to dEchorate hdf5 dataset", type=str)
    args = parser.parse_args()

    path_to_hdf = Path(args.hdf)
    dset = h5py.File(path_to_hdf, mode='r')

    signals = list(dset.keys())
    signal = signals[0]
    print('Signals:\t', signals)

    rooms = list(dset[signals[0]].keys())
    print('Rooms:\t', rooms)

    sources = list(dset[signals[0]][rooms[0]].keys())
    print('Sources:\t', sources)
    sources_description = ['Directional Avanton MIXCUBE']*6 + ['Omnidirectional B&G']*3
    sources_views = np.array([
        [ 0,-1, 0], # dir1
        [ 0, 1, 0], # dir2
        [-1, 0, 0], # dir3
        [-1, 0, 0], # dir4
        [ 0,-1, 0], # inv-dir5
        [ 1, 0, 0], # inv-dir6
        [ 0, 0, 0], # omni1
        [ 0, 0, 0], # omni2
        [ 0, 0, 0]  # omni3
    ])

    path_to_db = Path(args.csv)
    df = pd.read_csv(path_to_db)

    ### RIR MATRIX ###

    print('Building RIR matrix...', end='')
    D = len(rooms)
    J = len(sources)
    I = 30
    Fs = constants['Fs']
    L = int(2*Fs)

    rirs = np.zeros([L, I, J, D])

    for r, room in enumerate(rooms):

        for j, src in enumerate(sources):

            group = f'/{signal}/{room}/{src}'
            data = np.asarray(dset[group])
            
            rirs[:,:,j,r] = data[:2*Fs,:-1]
    
    print(' Done.')
    print('RIR matrix shape', rirs.shape)

    ### ECHO ANNOTATION ### 

    path_to_echo_df = Path(args.toa)
    echo_df = pd.read_csv(path_to_echo_df)

    ### RIRs to SOFAR's DRIRS

    for room_idx in tqdm(range(D), desc='room'):

        for src_idx in tqdm(range(J), desc='src'):
            
            for arr_idx in range(I // 5):

                sofa = sf.Sofa('SingleRoomSRIR')

                # get rirs
                mic_idxs = np.arange(5*arr_idx,5*(arr_idx+1))
                mimo_srir = rirs[:, mic_idxs, src_idx, room_idx]

                curr_df = df.loc[
                        (df['room_code'] == int(rooms[room_idx]))
                    &   (df['src_id'] == src_idx+1)
                    &   (df['array_id'] == arr_idx+1)
                    &   (df['src_signal'] == 'chirp')
                ]
                assert len(curr_df) == 5

                # source pos
                src_x = curr_df['src_pos_x'].unique()
                src_y = curr_df['src_pos_y'].unique()
                src_z = curr_df['src_pos_z'].unique()
                assert len(src_x) == len(src_y) == len(src_z) == 1
                src_pos = [[src_x, src_y, src_z]]

                # array pos
                arr_x = curr_df['array_bar_x'].unique()
                arr_y = curr_df['array_bar_y'].unique()
                arr_z = curr_df['array_bar_z'].unique()
                assert len(arr_x) == len(arr_y) == len(arr_z) == 1
                arr_pos = [[arr_x, arr_y, arr_z]]

                # mics pos
                mics_x = np.array(curr_df['mic_pos_x'])[None,:]
                mics_y = np.array(curr_df['mic_pos_y'])[None,:]
                mics_z = np.array(curr_df['mic_pos_z'])[None,:]
                mics_pos = np.concatenate([mics_x, mics_y, mics_z], axis=0)
                slope, intercept, r_value, p_value, std_err = stats.linregress(mics_x, mics_y)
                view = np.array([-slope, 1])
                view = view / np.linalg.norm(view)
                array_view = np.array([view[0], view[1], 0])
                # array_view = [1,0,0]

                # echo note
                if src_idx > 3:
                    toas = np.zeros((7,5))
                else:
                    curr_echo_df = echo_df.loc[(echo_df['arr'] == arr_idx) & (echo_df['src'] == src_idx)]
                    assert len(curr_echo_df) == 5
                    echo_toa_list = [
                        np.array(curr_echo_df['direct'] )[None,:],
                        np.array(curr_echo_df['floor']  )[None,:],
                        np.array(curr_echo_df['ceiling'])[None,:],
                        np.array(curr_echo_df['west']   )[None,:],
                        np.array(curr_echo_df['south']  )[None,:],
                        np.array(curr_echo_df['east']   )[None,:],
                        np.array(curr_echo_df['north']  )[None,:]]
                    toas = np.concatenate(echo_toa_list, axis=0)

                # rec filename
                filename = curr_df['filename'].unique()[0]

                room_rfl = {
                    'f' : 'True' if curr_df['room_rfl_floor'].unique()[0] > 0 else 'False',
                    'c' : 'True' if curr_df['room_rfl_ceiling'].unique()[0] > 0 else 'False',
                    'w' : 'True' if curr_df['room_rfl_west'].unique()[0]  > 0 else 'False',
                    's' : 'True' if curr_df['room_rfl_south'].unique()[0] else 'False',
                    'e' : 'True' if curr_df['room_rfl_east'].unique()[0]  else 'False',
                    'n' : 'True' if curr_df['room_rfl_north'].unique()[0] else 'False',
                    'frn':'True' if curr_df['room_fornitures'].unique()[0] else 'False',
                }

                
                sofa.GLOBAL_DatabaseName = 'dEchorate'

                title_suffix = f'room:{rooms[room_idx]}_src:{src_idx+1}_arr:{arr_idx+1}_mics:{mic_idxs.min()+1}-{mic_idxs.max()+1}'
                sofa.GLOBAL_Title = f'dEchorate_{title_suffix}'

                sofa.GLOBAL_DateCreated = '2022-04-20'
                sofa.GLOBAL_DateModified = sofa.GLOBAL_DateCreated

                sofa.GLOBAL_AuthorContact = 'diego.dicarlo89@gmail.com'
                sofa.GLOBAL_Organization = 'INRIA Rennes (Fr)'
                sofa.GLOBAL_License = 'The MIT License (MIT) Copyright (c) 2019, Diego Di Carlo. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.'
                sofa.GLOBAL_ApplicationName = 'Python'
                sofa.GLOBAL_ApplicationVersion = platform.python_version()


                sofa.GLOBAL_Comment = """- website: 'http://diegodicarlo.com/datasets/dechorate';
                - data: 'https://zenodo.org/record/5562386';
                - code: 'https://github.com/Chutlhu/DechorateDB'; 
                - paper: 'https://rdcu.be/cLJIG'"""

                sofa.GLOBAL_RoomDescription = 'Recorded at the Bar\'Ilan acoustic lab. Refer to the publication for all the details.'

                sofa.GLOBAL_History = """
                - 2019-01-20: Recorded raw data
                - 2019-06-17: Processed annotation
                - 2021-03-22: Publication on Zenodo v0
                - 2022-04-21: Creation SOFA"""


                sofa.GLOBAL_References = """@article{dicarlo2021dechorate,
                title={dEchorate: a calibrated room impulse response dataset for echo-aware signal processing},
                author={Di Carlo, Diego and Tandeitnik, Pinchas and Foy, Cedri{\'c} and Bertin, Nancy and Deleforge, Antoine and Gannot, Sharon},
                journal={EURASIP Journal on Audio, Speech, and Music Processing},
                volume={2021},
                number={1},
                pages={1--15},
                year={2021},
                publisher={Springer}
                }"""

                ## --- RIRS --- ##
                sofa.Data_IR = mimo_srir.T[None,:,:] # must be in M x R x N x E (measurements, receivers, samples, emitters)
                assert sofa.Data_IR.shape == (1,5,2*Fs)
                sofa.Data_SamplingRate = Fs
                sofa.Data_SamplingRate_Units = 'hertz'
                sofa.Data_Delay = [0] * 5

                ## --- RECORDINGS --- ##
                sofa.add_attribute("GLOBAL_RecordingFilenames", filename)

                ## --- ARRAY --- ##

                # Listener = compact microphone array (it may contain many receiver, i.e. mics)
                sofa.GLOBAL_ListenerShortName = f'array_id:{arr_idx}'

                sofa.GLOBAL_ListenerDescription = 'handcrafted 5-mics non-Uniform Linear Array'

                sofa.ListenerPosition = arr_pos
                sofa.ListenerPosition_Type = 'cartesian'
                sofa.ListenerPosition_Units = 'metre'
                sofa.ListenerView = array_view
                sofa.ListenerUp = [0,0,1]
                sofa.ListenerView_Type = 'cartesian'
                sofa.ListenerView_Units = 'metre'

                ## --- MICROPHONES --- ##

                # Receiver = MICROPHONE
                sofa.GLOBAL_ReceiverShortName = 'capsule'
                sofa.GLOBAL_ReceiverDescription = 'AKG CK32'
                sofa.ReceiverPosition = mics_pos.T
                sofa.ReceiverPosition_Type = 'cartesian'
                sofa.ReceiverPosition_Units = 'metre'
                sofa.ReceiverView = [[1,0,0]]*5
                sofa.ReceiverUp = [[0,0,1]]*5
                sofa.ReceiverView_Type = 'cartesian'
                sofa.ReceiverView_Units = 'metre'

                # sofa.GLOBAL_EmitterShortName = sofa.GLOBAL_SourceShortName
                # # print(sofa.GLOBAL_EmitterShortName)
                # sofa.GLOBAL_EmitterDescription = sofa.GLOBAL_SourceDescription
                # # print(sofa.GLOBAL_EmitterDescription)

                # sofa.EmitterPosition = sofa.SourcePosition
                # # print(sofa.EmitterPosition)
                # sofa.EmitterPosition_Type = sofa.SourcePosition_Type
                # # print(sofa.EmitterPosition_Type)
                # sofa.EmitterPosition_Units = sofa.SourcePosition_Units
                # # print(sofa.EmitterPosition_Units)
                # sofa.EmitterView = sofa.SourceView
                # # print(sofa.EmitterView)
                # sofa.EmitterUp = sofa.SourceUp
                # # print(sofa.EmitterUp)
                # sofa.EmitterView_Type = sofa.EmitterView_Type
                # # print(sofa.EmitterView_Type)
                # sofa.EmitterView_Units = sofa.EmitterView_Units
                # # print(sofa.EmitterView_Units)

                # --- SOURCES --- ## 

                # Source = compact sound source (it may contain more emitters, i.e. speakers)
                sofa.GLOBAL_SourceShortName = f'loudspeaker_id:{src_idx}'
                sofa.GLOBAL_SourceDescription = sources_description[src_idx]
                sofa.SourcePosition = src_pos
                sofa.SourcePosition_Type = 'cartesian'
                sofa.SourcePosition_Units = 'metre'
                sofa.SourceView = sources_views[src_idx,:]
                sofa.SourceUp = [0,0,1]
                sofa.SourceView_Type = 'cartesian'
                sofa.SourceView_Units = 'metre'

                ## --- ROOM --- ##
                sofa.GLOBAL_RoomType = 'shoebox'
                sofa.RoomCornerA = [0,0,0]
                sofa.RoomCornerB = constants['room_size']
                sofa.RoomCorners_Type = 'cartesian'
                sofa.RoomCorners_Units = 'metre'


                sofa.add_variable("ReflectiveSurfacesCode", rooms[room_idx], "string", "S")
                sofa.add_attribute("ReflectiveSurfacesCode_Order", "Floor,West,South,East,North")

                sofa.add_variable("RoomFloorIsRefective",    [room_rfl['f']], "string", "S")
                sofa.add_variable("RoomCleilingIsRefective", [room_rfl['c']], "string", "S")
                sofa.add_variable("RoomWestIsRefective",     [room_rfl['w']], "string", "S")
                sofa.add_variable("RoomSouthIsRefective",    [room_rfl['s']], "string", "S")
                sofa.add_variable("RoomEastIsRefective",     [room_rfl['e']], "string", "S")
                sofa.add_variable("RoomNorthIsRefective",    [room_rfl['n']], "string", "S")
                sofa.add_variable("RoomHasFornitures",       [room_rfl['frn']], "string", "S")

                sofa.add_variable("Temperature", constants['room_temperature'], "double", "I")
                sofa.add_attribute("Temperature_Units", "degree Celsius")
                sofa.add_variable("RelativeHumidity", constants['room_humidity'], "double", "I")
                sofa.add_attribute("RelativeHumidity_Units", "percentage")
                sofa.add_variable("SpeedOfSound", constants['speed_of_sound'], "double", "I")
                sofa.add_attribute("SpeedOfSound_Units", "metre/seconds")
                
                ## --- ECHOES --- ##
                sofa.add_variable("EchoTimingsPeakPicking", toas, "double", "KR")
                sofa.add_attribute("EchoTimingsPeakPicking_Units", "samples")

                sofa.verify()

                filename = sofa.GLOBAL_Title
                path_to_out = Path(f'./outputs/dEchorate_sofa/{filename}.sofa')
                if not path_to_out.exists():
                    sf.write_sofa(str(path_to_out), sofa, compression=7)