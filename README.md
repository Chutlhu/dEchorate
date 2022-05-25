# WELCOME to the dECHORATE dataset

## dECHORATE: A calibrated dataset for echo-aware audio signal processing

> The `dEchorate` dataset is a new database of measured multichannel Room Impulse Responses (RIRs) including annotations of early echo timings and 3D positions of microphones, real sources and image sourcesunder different wall configurations in a cuboid room.  
> These data provide a tool for benchmarking recentmethods inecho-awarespeech enhancement, room geometry estimation, RIR estimation, acoustic echoretrieval, microphone calibration, echo labeling and reflectors estimation.  
> The database is accompanied withsoftware utilities to easily access, manipulate and visualize the data as well as baseline methods forecho-related tasks.  

**Keywords:** Echo-aware signal processing; Acoustic echoes; Room impulse response; Audio database; AcousticEcho Retrieval; Spatial Filtering; Room Geometry Estimation; Microphone arrays.

You can find a detailed explanation of the dEchorate dataset at:
[dEchorate: a Calibrated Room Impulse Response Database for Echo-aware Signal Processing](https://hal.archives-ouvertes.fr/hal-03207860/)

`dEchorate` has three main elements:
- the data on Zenodo
    - [dataset in HDF5 format with python script](http://zenodo.org/record/6576203);
    - [dataset in SOFA format](http://zenodo.org/record/6580691);
- the code for working with `dEchorate` (available in this repo);
- the code for reproducing the [paper](https://hal.archives-ouvertes.fr/hal-03207860/) (available in this repo).

## News:
- 25 May 2022: v2.0.0 new dEchorate dataset
- 03 May 2020: v0.0.1 dEchorate project is alive

<!-- ## Get the Data
The data is available at [Zenodo](https://zenodo.org/record/6576203).
Please, follow that link to download (part of) the data.
The dataset is available in multiple ways:
- Annotations/labels/metadata: csv file that can be used pandas (Python)
- Only RIRs: numpy matrix `n_samples x n_mics x n_srcs x n_rooms` (~ 6 GB)
- Only Speech: numpy matrix `n_samples x n_mics x n_srcs x n_rooms` (~ 6 GB)
- Raw data: an hdf5 file (~ 75 GB) containig all the raw recording data (chirps, speech/noise sources, babble noise, room tone) -->

## Citing dEchorate

If you are using `dEchorate` and you want to cite us, please use

```BibTex
@article{dicarlo2021dechorate,
  title={dEchorate: a calibrated room impulse response dataset for echo-aware signal processing},
  author={{Di Carlo}, Diego and Tandeitnik, Pinchas and Foy, Cedri{\'c} and Bertin, Nancy and Deleforge, Antoine and Gannot, Sharon},
  journal={EURASIP Journal on Audio, Speech, and Music Processing},
  volume={2021},
  number={1},
  pages={1--15},
  year={2021},
  publisher={Springer}
}
```
