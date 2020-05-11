import numpy as np
import pandas as pd

from tqdm import tqdm

# load csv with objects position
path_to_positioning_note = './data/dECHORATE/annotations/dECHORATE_positioning_note.csv'
pos_note_df = pd.read_csv(path_to_positioning_note)
# load csv with recordings annotation
path_to_recordings_note = './data/dECHORATE/annotations/dECHORATE_recordings_note.csv'
rec_note_df = pd.read_csv(path_to_recordings_note)

df = pd.DataFrame()
c = 0

for r, row in tqdm(rec_note_df.iterrows()):

    for i in range(31):

        df.at[c, 'filename'] = row['filename']
        df.at[c, 'src_id'] = row['id']
        df.at[c, 'src_ch'] = row['channel']

        if row['sources'] == 'silence':
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
                1/0
            df.at[c, 'src_pos_x'] = float(curr_pos_source['x'].values)
            df.at[c, 'src_pos_y'] = float(curr_pos_source['y'].values)
            df.at[c, 'src_pos_z'] = float(curr_pos_source['z'].values)

        df.at[c, 'src_signal'] = row['signal']
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
        try:
            assert len(curr_pos_array) == 1
        except:
            print('array', i//5+1, i)
            print(curr_pos_array)
            1/0

        df.at[c, 'array_bar_x'] = float(curr_pos_array['x'].values)
        df.at[c, 'array_bar_y'] = float(curr_pos_array['y'].values)
        df.at[c, 'array_bar_z'] = float(curr_pos_array['z'].values)
        df.at[c, 'array_bar_theta'] = curr_pos_array['theta'].values
        df.at[c, 'array_bar_aiming_at_corner'] = curr_pos_array['aiming_at'].values

        # find mic attributes in pos_note
        curr_pos_mic = pos_note_df.loc[
            (pos_note_df['type'] == 'mic')
            & (pos_note_df['id'] == i+1)]
        try:
            assert len(curr_pos_mic) == 1
        except:
            print('mic', i)
            print(curr_pos_mic)
            1/0


        df.at[c, 'mic_id'] = curr_pos_mic['id'].values
        df.at[c, 'mic_ch'] = curr_pos_mic['channel'].values
        df.at[c, 'mic_pos_x'] = float(curr_pos_mic['x'].values)
        df.at[c, 'mic_pos_y'] = float(curr_pos_mic['y'].values)
        df.at[c, 'mic_pos_z'] = float(curr_pos_mic['z'].values)
        df.at[c, 'mic_signal'] = row['signal']

        c += 1

    path_to_database = './data/dECHORATE/annotations/dECHORATE_database.csv'
    df.to_csv(path_to_database)
