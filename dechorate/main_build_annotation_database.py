import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", help="Path to output files", type=str)
    parser.add_argument("--datadir", help="Path to dEchorate raw dataset", type=str)
    parser.add_argument("--calibnote", help="Path to calibrated position database", type=str)
    
    args = parser.parse_args()

    output_dir = Path(args.outdir)
    assert output_dir.exists()

    path_to_calib = Path(args.calibnote)
    assert path_to_calib.exists()

    dataset_dir = Path(args.datadir)
    assert dataset_dir.exists()

    # load csv with position annotation
    df_pos = pd.read_csv(path_to_calib)
    df_pos["id"]  = pd.to_numeric(df_pos["id"], errors='coerce')
    df_pos["channel"]  = pd.to_numeric(df_pos["channel"], errors='coerce')

    # load csv with recordings annotation
    path_to_recordings_db = dataset_dir / Path('dEchorate_recordings_annotation.csv')
    df_rec = pd.read_csv(path_to_recordings_db)
    df_rec["id"]  = pd.to_numeric(df_rec["id"], errors='coerce')
    df_rec["channel"]  = pd.to_numeric(df_rec["channel"], errors='coerce')

    # path to output file
    path_to_output_database = output_dir / Path('dEchorate_database.csv')

    # initialize database
    df = pd.DataFrame({
        'filename': pd.Series(dtype='str'),
        'src_id': pd.Series(dtype='int'),
        'src_ch': pd.Series(dtype='int'),
        'src_type' : pd.Series(dtype='str'),
        'src_signal' : pd.Series(dtype='str'),
        'src_pos_x': pd.Series(dtype='float'),
        'src_pos_y': pd.Series(dtype='float'),
        'src_pos_z': pd.Series(dtype='float'),
        'room_code'  : pd.Series(dtype='str'),
        'room_rfl_floor' : pd.Series(dtype='int'),
        'room_rfl_ceiling' : pd.Series(dtype='int'),
        'room_rfl_west' : pd.Series(dtype='int'),
        'room_rfl_south' : pd.Series(dtype='int'),
        'room_rfl_east' : pd.Series(dtype='int'),
        'room_rfl_north' : pd.Series(dtype='int'),
        'room_fornitures' : pd.Series(dtype='bool'),
        'room_temperature' : pd.Series(dtype='float'),
        'rec_silence_dB' : pd.Series(dtype='float'),
        'rec_artifacts' : pd.Series(dtype='int'),
        'mic_type' : pd.Series(dtype='str'),
        'mic_id' : pd.Series(dtype='int'),
        'mic_ch' : pd.Series(dtype='int'),
        'mic_pos_x' : pd.Series(dtype='float'),
        'mic_pos_y' : pd.Series(dtype='float'),
        'mic_pos_z' : pd.Series(dtype='float'),
        'array_id' : pd.Series(dtype='int'),
        'array_bar_x' : pd.Series(dtype='float'),
        'array_bar_x' : pd.Series(dtype='float'),
        'array_bar_y' : pd.Series(dtype='float'),
        'array_bar_z' : pd.Series(dtype='float'),
    })


    c = 0  # counter

    print('Compiling an unique database')
    # for each recording
    for r, row in tqdm(df_rec.iterrows(), total=len(df_rec)):

        # for each channel in the recordings
        for i in range(31):

            src_id = row['id']

            df.at[c, 'filename'] = row['filename']
            df.at[c, 'src_id'] = src_id
            df.at[c, 'src_ch'] = row['channel']


            # if silence or diffuse (=noise) skip
            if row['sources'] == 'silence':
                df.at[c, 'src_pos_x'] = np.nan
                df.at[c, 'src_pos_y'] = np.nan
                df.at[c, 'src_pos_z'] = np.nan

            else:
                # find src attributes in pos_note
                curr_pos_source = df_pos.loc[
                    (df_pos['type'] == row['sources'])
                    & (df_pos['channel'] == row['channel'])
                    & (df_pos['id'] == src_id)]
                
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

                df.at[c, 'src_pos_x'] = float(curr_pos_source['x'].values[0])
                df.at[c, 'src_pos_y'] = float(curr_pos_source['y'].values[0])
                df.at[c, 'src_pos_z'] = float(curr_pos_source['z'].values[0])

                df.at[c, 'src_view_x'] = float(curr_pos_source['view_x'].values[0])
                df.at[c, 'src_view_y'] = float(curr_pos_source['view_y'].values[0])
                df.at[c, 'src_view_z'] = float(curr_pos_source['view_z'].values[0])


            if row['id'] >= 4 and row['sources'] == 'directional':
                df.at[c, 'src_type'] = 'invdirectional'
            else:
                df.at[c, 'src_type'] = row['sources']

            df.at[c, 'src_signal'] = row['signal']
            df.at[c, 'room_code'] = '%d%d%d%d%d%d' % (row['floor'], row['ceiling'], row['west'], row['south'], row['east'], row['north'])
            df.at[c, 'room_rfl_floor'] = row['floor']
            df.at[c, 'room_rfl_ceiling'] = row['ceiling']
            df.at[c, 'room_rfl_west'] = row['west']
            df.at[c, 'room_rfl_south'] = row['south']
            df.at[c, 'room_rfl_east'] = row['east']
            df.at[c, 'room_rfl_north'] = row['north']
            df.at[c, 'room_fornitures'] = row['fornitures']
            if row['fornitures']:
                df.at[c, 'room_code'] = '020002'

            df.at[c, 'room_temperature'] = row['temperature']
            df.at[c, 'rec_silence_dB'] = row['silence dB']
            df.at[c, 'rec_artifacts'] = row['artifacts']

            # find array attributes in pos_note
            if i == 30:
                df.at[c, 'mic_type'] = 'loopback'
                df.at[c, 'mic_id'] = i

            else:
                df.at[c, 'mic_type'] = 'capsule'

                curr_pos_array = df_pos.loc[
                    (df_pos['type'] == 'array')
                    & (df_pos['id'] == i//5)]

                # check that current selection is unique
                try:
                    assert len(curr_pos_array) == 1
                except:
                    print('array', i//5, i)
                    print(curr_pos_array)
                    ValueError('Too many arrays')

                df.at[c, 'array_id'] = i//5
                df.at[c, 'array_bar_pos_x'] = float(curr_pos_array['x'].values[0])
                df.at[c, 'array_bar_pos_y'] = float(curr_pos_array['y'].values[0])
                df.at[c, 'array_bar_pos_z'] = float(curr_pos_array['z'].values[0])

                df.at[c, 'array_bar_view_x'] = float(curr_pos_array['view_x'].values[0])
                df.at[c, 'array_bar_view_y'] = float(curr_pos_array['view_y'].values[0])
                df.at[c, 'array_bar_view_z'] = float(curr_pos_array['view_z'].values[0])

                # find mic attributes in pos_note
                curr_pos_mic = df_pos.loc[
                    (df_pos['type'] == 'mic')
                    & (df_pos['id'] == i)]

                # check that current selection is unique
                try:
                    assert len(curr_pos_mic) == 1
                except:
                    print('mic', i)
                    print(curr_pos_mic)
                    raise ValueError('Too many microphones')

                curr_mic = int(curr_pos_mic['id'].values[0])
                df.at[c, 'mic_id'] = int(curr_pos_mic['id'].values[0])
                df.at[c, 'mic_ch'] = int(curr_pos_mic['channel'].values[0])
                df.at[c, 'mic_pos_x'] = float(curr_pos_mic['x'].values[0])
                df.at[c, 'mic_pos_y'] = float(curr_pos_mic['y'].values[0])
                df.at[c, 'mic_pos_z'] = float(curr_pos_mic['z'].values[0])
                df.at[c, 'mic_view_x'] = float(curr_pos_mic['view_x'].values[0])
                df.at[c, 'mic_view_y'] = float(curr_pos_mic['view_y'].values[0])
                df.at[c, 'mic_view_z'] = float(curr_pos_mic['view_z'].values[0])

            c += 1

        if c % 100 == 0:
            df.to_csv(path_to_output_database)

    df.to_csv(path_to_output_database)

    print('done.')
    print('You can find the current database in:\n%s' % path_to_output_database)
