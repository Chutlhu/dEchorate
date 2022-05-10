# Paths
path_to_data_dir="/home/chutlhu/Documents/Datasets/dEchorate/raw/"


# # 1. annotation files to database
python dechorate/main_geometry_from_measurements.py $path_to_data_dir

# # 2. Final calibrated geometry
python dechorate/main_geometry_from_echo_calibration.py $path_to_data_dir

# # 3. Build the complete database with recordings 
path_to_calibrated_positions_notes='outputs/dEchorate_calibrated_elements_positions.csv'
python dechorate/main_build_annotation_database.py $path_to_data_dir $path_to_calibrated_positions_notes

# # 4. Build sound dataset: from zips to hdf5
path_to_database="outputs/dEchorate_database.csv"
python dechorate/main_build_sound_datasets.py --signal silence --datadir $path_to_data_dir --dbpath $path_to_database
python dechorate/main_build_sound_datasets.py --signal babble  --datadir $path_to_data_dir --dbpath $path_to_database
python dechorate/main_build_sound_datasets.py --signal noise   --datadir $path_to_data_dir --dbpath $path_to_database
python dechorate/main_build_sound_datasets.py --signal speech  --datadir $path_to_data_dir --dbpath $path_to_database
python dechorate/main_build_sound_datasets.py --signal chirp   --datadir $path_to_data_dir --dbpath $path_to_database
echo "you may want to delete the content of .cache folder" 

# # # 5. Estimate RIRs
# path_to_chirps="outputs/dEchorate_chirp.hdf5"
# python dechorate/main_estimate_rirs.py --dbpath $path_to_database --chirps $path_to_chirps


# # # 6. Repack datasets
# for signal in silence babble noise speech chirp rirs; do
#     h5repack -v -f GZIP=7 outputs/dEchorate_${signal}.hdf5 outputs/dEchorate_${signal}_gzip7.hdf5
# done