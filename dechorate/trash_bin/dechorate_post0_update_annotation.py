import os
import numpy as np
import pandas as pd

from tqdm import tqdm

from dechorate.exception import NotUniqueSelectionError
from dechorate.utils.file_utils import load_from_pickle

# path to output folder
path_to_output_database = os.path.join('data','final','manual_annatotion.csv')
path_to_output_database = os.path.join('data','final','annatotion.csv')

# load csv with objects position annotation
path_to_positioning_note = os.path.join('data', 'dECHORATE', 'annotations', 'manual_annotation_dECHORATE_positioning.csv')
pos_note_df = pd.read_csv(path_to_positioning_note)

# load csv with recordings annotation
path_to_recordings_note = os.path.join('data', 'dECHORATE', 'annotations', 'manual_annotation_dECHORATE_recordings.csv')
rec_note_df = pd.read_csv(path_to_recordings_note)

# load processed data after calibration
path_to_note_position_pkl = os.path.join('data', 'final', 'mics_srcs_pos.pkl')
mics_srcs_pos = load_from_pickle(path_to_note_position_pkl)

# initialize database
df = pd.DataFrame()
c = 0  # counter

print('Compiling an unique database')

for r, row in tqdm(rec_note_df.iterrows()):

    for i in range(31):

        df.at[c, 'filename'] = row['filename']
        df.at[c, 'src_id'] = row['id']
        df.at[c, 'src_ch'] = row['channel']

        if row['sources'] == 'silence' or row['sources'] == 'babble':
            pass

        else:
            # find src attributes in pos_note
            curr_pos_source = pos_note_df.loc[
              (pos_note_df['type'] == row['sources'])
            & (pos_note_df['channel'] == row['channel'])
            & (pos_note_df['id'] == row['id'])]
            try:
                assert len(curr_pos_source) == 1
            except:
                print(row)
                print(row['sources'], row['channel'], row['id'])
                print(curr_pos_source)
                raise NotUniqueSelectionError('Too many sources')

            df.at[c, 'src_pos_x'] = float(curr_pos_source['x'].values)
            df.at[c, 'src_pos_y'] = float(curr_pos_source['y'].values)
            df.at[c, 'src_pos_z'] = float(curr_pos_source['z'].values)

            if row['id'] in [1, 2, 3, 4]:
                curr_pos_source_calib = mics_srcs_pos['srcs']
                df.at[c, 'src_pos_x_calib'] = float(mics_srcs_pos['srcs'][0, row['id']-1])
                df.at[c, 'src_pos_y_calib'] = float(mics_srcs_pos['srcs'][1, row['id']-1])
                df.at[c, 'src_pos_z_calib'] = float(mics_srcs_pos['srcs'][2, row['id']-1])

        df.at[c, 'src_signal'] = row['signal']
        df.at[c, 'room_code'] = '%d%d%d%d%d%d' % (
            row['floor'], row['ceiling'], row['west'], row['south'], row['east'], row['north'])
        df.at[c, 'room_rfl_floor'] = row['floor']
        df.at[c, 'room_rfl_ceiling'] = row['ceiling']
        df.at[c, 'room_rfl_west'] = row['west']
        df.at[c, 'room_rfl_south'] = row['south']
        df.at[c, 'room_rfl_east'] = row['east']
        df.at[c, 'room_rfl_north'] = row['north']
        df.at[c, 'room_fornitures'] = row['fornitures']

        df.at[c, 'room_temperature'] = row['temperature']
        df.at[c, 'rec_silence_dB'] = row['silence dB']
        df.at[c, 'rec_artifacts'] = row['artifacts']

        # find array attributes in pos_note
        if i == 30:
            df.at[c, 'mic_type'] = 'loopback'
            c += 1
            continue
        else:
            df.at[c, 'mic_type'] = 'capsule'

        curr_pos_array = pos_note_df.loc[
            (pos_note_df['type'] == 'array')
            & (pos_note_df['id'] == i//5+1)]

        # check that current selection is unique
        try:
            assert len(curr_pos_array) == 1
        except:
            print('array', i//5+1, i)
            print(curr_pos_array)
            NotUniqueSelectionError('Too many arrays')

        df.at[c, 'array_bar_x'] = float(curr_pos_array['x'].values)
        df.at[c, 'array_bar_y'] = float(curr_pos_array['y'].values)
        df.at[c, 'array_bar_z'] = float(curr_pos_array['z'].values)
        df.at[c, 'array_bar_theta'] = curr_pos_array['theta'].values
        df.at[c, 'array_bar_aiming_at_corner'] = curr_pos_array['aiming_at'].values

        # find mic attributes in pos_note
        curr_pos_mic = pos_note_df.loc[
            (pos_note_df['type'] == 'mic')
            & (pos_note_df['id'] == i+1)]

        # check that current selection is unique
        try:
            assert len(curr_pos_mic) == 1
        except:
            print('mic', i)
            print(curr_pos_mic)
            raise NotUniqueSelectionError('Too many microphones')

        curr_mic = int(curr_pos_mic['id'].values[0])
        df.at[c, 'mic_id'] = curr_pos_mic['id'].values
        df.at[c, 'mic_ch'] = curr_pos_mic['channel'].values
        df.at[c, 'mic_pos_x'] = float(curr_pos_mic['x'].values)
        df.at[c, 'mic_pos_y'] = float(curr_pos_mic['y'].values)
        df.at[c, 'mic_pos_z'] = float(curr_pos_mic['z'].values)
        df.at[c, 'mic_signal'] = row['signal']

        if curr_mic in range(1, 31):
            curr_pos_source_calib = mics_srcs_pos['mics']
            df.at[c, 'mic_pos_x_calib'] = float(mics_srcs_pos['mics'][0, curr_mic-1])
            df.at[c, 'mic_pos_y_calib'] = float(mics_srcs_pos['mics'][1, curr_mic-1])
            df.at[c, 'mic_pos_z_calib'] = float(mics_srcs_pos['mics'][2, curr_mic-1])

        c += 1

    if c % 100 == 0:
        df.to_csv(path_to_output_database)

df.to_csv(path_to_output_database)

print('done.')
print('You can find the current database in:\n%s' % path_to_output_database)
