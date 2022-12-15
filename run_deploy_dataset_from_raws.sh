# Paths
path_to_data_dir="/home/chutlhu/Documents/Datasets/dEchorate/raw/"
path_to_data_dir="/home/dicarlo_d/Documents/Datasets/dEchorate/BarIlan_data"
outdir="./outputs"

path_to_database="${outdir}/dEchorate_database.csv"
path_to_calibrated_positions_notes="${outdir}/dEchorate_calibrated_elements_positions.csv"
path_to_chirps="${outdir}/dEchorate_chirp.hdf5"

# # # # 1. annotation files to database
# python dechorate/main_geometry_from_measurements.py --outdir ${outdir} --datadir $path_to_data_dir

# # # # 2. Final calibrated geometry
# python dechorate/main_geometry_from_echo_calibration.py --outdir ${outdir}

# # # 3. Build the complete database with recordings 
# python dechorate/main_build_annotation_database.py --outdir ${outdir} --datadir $path_to_data_dir --calibnote $path_to_calibrated_positions_notes

# # # # 4. Build sound dataset: from zips to hdf5
# python dechorate/main_build_sound_datasets.py --outdir ${outdir} --signal silence --datadir $path_to_data_dir --dbpath $path_to_database --comp 7
# python dechorate/main_build_sound_datasets.py --outdir ${outdir} --signal babble  --datadir $path_to_data_dir --dbpath $path_to_database --comp 7
# python dechorate/main_build_sound_datasets.py --outdir ${outdir} --signal noise   --datadir $path_to_data_dir --dbpath $path_to_database --comp 7
# python dechorate/main_build_sound_datasets.py --outdir ${outdir} --signal speech  --datadir $path_to_data_dir --dbpath $path_to_database --comp 7
# python dechorate/main_build_sound_datasets.py --outdir ${outdir} --signal chirp   --datadir $path_to_data_dir --dbpath $path_to_database --comp 7
# echo "you may want to delete the content of .cache folder" 

# # # 5. Estimate RIRs
python dechorate/main_estimate_rirs.py --outdir ${outdir} --dbpath ${path_to_database} --chirps ${path_to_chirps} --comp 9

# # # # 6. Convert it into Sofa format
# python dechorate/main_build_sofa_database.py  --outdir "${outdir}/sofa/" --toa "$path_to_data_dir/echo_pickpeaking.csv" --csv "${outdir}/dEchorate_database.csv" --hdf "${outdir}/dEchorate_rirs.hdf5"

# # # 6. Repack datasets
# for signal in silence babble noise speech chirp rirs; do
#     h5repack -v -f GZIP=7 ${outdir}/dEchorate_${signal}.hdf5 ${outdir}/dEchorate_${signal}_gzip7.hdf5
# done