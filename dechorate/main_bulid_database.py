import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path


# load csv with position annotation
path_to_position_db = Path('data', 'dEchorate_calibrated_elements_positions.csv')
df_pos = pd.read_csv(path_to_position_db)
df_pos["id"]  = pd.to_numeric(df_pos["id"], errors='coerce')
df_pos["channel"]  = pd.to_numeric(df_pos["channel"], errors='coerce')

# load csv with recordings annotation
path_to_recordings_db = Path('data', 'dEchorate_recordings_annotation.csv')
df_rec = pd.read_csv(path_to_recordings_db)
df_rec["id"]  = pd.to_numeric(df_rec["id"], errors='coerce')
df_rec["channel"]  = pd.to_numeric(df_rec["channel"], errors='coerce')

# path to output file
path_to_output_database = Path('data','dEchorate_database.csv')

# initialize database
df = pd.DataFrame()
c = 0  # counter

print('Compiling an unique database')
# for each recording
for r, row in tqdm(df_rec.iterrows()):

    # for each channel in the recordings
    for i in range(31):

        df.at[c, 'filename'] = row['filename']
        df.at[c, 'src_id'] = row['id']
        df.at[c, 'src_ch'] = row['channel']

        # if silence or diffuse (=noise) skip
        if not (row['sources'] == 'silence' or row['sources'] == 'diffuse'):

            # find src attributes in pos_note
            curr_pos_source = df_pos.loc[
                  (df_pos['type'] == row['sources'])
                & (df_pos['channel'] == row['channel'])
                & (df_pos['id'] == row['id'])]
            
            try:
                assert len(curr_pos_source) == 1

            except:
                print(row)
                print(row['sources'], row['channel'], row['id'])
                            
                print(df_pos.loc[(df_pos['type'] == row['sources'])])
                print(df_pos.loc[(df_pos['channel'] == row['channel'])])
                print(df_pos.loc[(df_pos['id'] == row['id'])])
                print(df_pos.loc[(df_pos['type'] == row['sources']) & (df_pos['channel'] == row['channel'])])
                print(df_pos.loc[(df_pos['type'] == row['sources']) & (df_pos['channel'] == row['channel']) & (df_pos['id'] == row['id'])])

                raise ValueError('Too many sources')

            df.at[c, 'src_pos_x'] = float(curr_pos_source['x'].values)
            df.at[c, 'src_pos_y'] = float(curr_pos_source['y'].values)
            df.at[c, 'src_pos_z'] = float(curr_pos_source['z'].values)

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

        else:
            df.at[c, 'mic_type'] = 'capsule'

            curr_pos_array = df_pos.loc[
                  (df_pos['type'] == 'array')
                & (df_pos['id'] == i//5+1)]

            # check that current selection is unique
            try:
                assert len(curr_pos_array) == 1
            except:
                print('array', i//5+1, i)
                print(curr_pos_array)
                ValueError('Too many arrays')

            df.at[c, 'array_bar_x'] = float(curr_pos_array['x'].values)
            df.at[c, 'array_bar_y'] = float(curr_pos_array['y'].values)
            df.at[c, 'array_bar_z'] = float(curr_pos_array['z'].values)

            # find mic attributes in pos_note
            curr_pos_mic = df_pos.loc[
                (df_pos['type'] == 'mic')
                & (df_pos['id'] == i+1)]

            # check that current selection is unique
            try:
                assert len(curr_pos_mic) == 1
            except:
                print('mic', i)
                print(curr_pos_mic)
                raise ValueError('Too many microphones')

            curr_mic = int(curr_pos_mic['id'].values[0])
            df.at[c, 'mic_id'] = curr_pos_mic['id'].values
            df.at[c, 'mic_ch'] = curr_pos_mic['channel'].values
            df.at[c, 'mic_pos_x'] = float(curr_pos_mic['x'].values)
            df.at[c, 'mic_pos_y'] = float(curr_pos_mic['y'].values)
            df.at[c, 'mic_pos_z'] = float(curr_pos_mic['z'].values)
            df.at[c, 'mic_signal'] = row['signal']

        c += 1

    if c % 100 == 0:
        df.to_csv(path_to_output_database)

df.to_csv(path_to_output_database)

print('done.')
print('You can find the current database in:\n%s' % path_to_output_database)
