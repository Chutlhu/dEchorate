#!/usr/bin/env bash

# Paths
path_to_data_dir="/home/dicarlod/Documents/Dataset/dEchorate"
outdir="./outputs_16k"

mkdir -p $outdir

path_to_database="${outdir}/dEchorate_database.csv"
path_to_calibrated_positions_notes="${outdir}/dEchorate_calibrated_elements_positions.csv"
path_to_chirps="${outdir}/dEchorate_chirp.h5"

# # 1. annotation files to database
python dechorate/main_geometry_from_measurements.py --outdir ${outdir} --datadir $path_to_data_dir

# # 2. Final calibrated geometry
python dechorate/main_geometry_from_echo_calibration.py --outdir ${outdir}

# # 3. Build the complete database with recordings 
python dechorate/main_build_annotation_database.py --outdir ${outdir} --datadir $path_to_data_dir --calibnote $path_to_calibrated_positions_notes

# # 4. Build sound dataset: from zips to hdf5
python dechorate/main_build_sound_datasets.py --outdir ${outdir} --signal silence --fs 16000 --datadir $path_to_data_dir --dbpath $path_to_database --comp 7
python dechorate/main_build_sound_datasets.py --outdir ${outdir} --signal babble  --fs 16000 --datadir $path_to_data_dir --dbpath $path_to_database --comp 7
python dechorate/main_build_sound_datasets.py --outdir ${outdir} --signal noise   --fs 16000 --datadir $path_to_data_dir --dbpath $path_to_database --comp 7
python dechorate/main_build_sound_datasets.py --outdir ${outdir} --signal speech  --fs 16000 --datadir $path_to_data_dir --dbpath $path_to_database --comp 7
python dechorate/main_build_sound_datasets.py --outdir ${outdir} --signal chirp   --fs 48000 --datadir $path_to_data_dir --dbpath $path_to_database --comp 7
echo "you may want to delete the content of .cache folder"

# 5. Estimate RIRs
# python dechorate/main_estimate_rirs.py --outdir ${outdir} --dbpath ${path_to_database} --chirps ${path_to_chirps} --comp 7

# 6. Convert it into Sofa format
# mkdir -p "${outdir}/sofa/"
# cp "$path_to_data_dir/echo_pickpeaking.csv" "${outdir}/dEchorate_echo_manual.csv" 
# python dechorate/main_build_sofa_database.py  --outdir "${outdir}/sofa/" --toa "$path_to_data_dir/echo_pickpeaking.csv" --csv "${outdir}/dEchorate_database.csv" --hdf "${outdir}/dEchorate_rirs.hdf5"

# 6. Repack datasets in different compression, if needed
# for signal in silence babble noise speech chirp rirs; do
#     h5repack -v -f GZIP=9 ${outdir}/dEchorate_${signal}.hdf5 ${outdir}/dEchorate_${signal}_gzip7.hdf5
# done

# 7. Preprare deliverable
# delivdir='./deliverable'
# mkdir -p $delivdir
# for signal in silence babble noise speech chirp rirs; do
#     cp ${outdir}/dEchorate_${signal}_gzip7.hdf5 ${delivdir}/dEchorate_${signal}.h5
# done
# cp ${outdir}/dEchorate_database.hdf5    ${delivdir}/dEchorate_database.h5
# cp ${outdir}/dEchorate_echo_notes.hdf5  ${delivdir}/dEchorate_annotations.h5
# cp ./dechorate/main_geometry_from_echo_calibration.py ${delivdir}/main_geometry_from_echo_calibration.py
