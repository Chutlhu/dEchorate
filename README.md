# WELCOME to the dECHORATE dataset

> :warning: **This work is under review**: Be very careful here!

## dECHORATE: A calibrated dataset for echo-aware audio signal processing

You can find a detailed explanation of the dEchorate dataset at:
[dEchorate: a Calibrated Room Impulse Response Database for Echo-aware Signal Processing](https://hal.archives-ouvertes.fr/hal-03207860/)

`dEchorate` has three main elements:
- the data (available on Zenodo);
- the code for working with `dEchorate`;
- the code for reproducing the [paper](https://hal.archives-ouvertes.fr/hal-03207860/).

## Contents
- [News](#news)
- [Get the Data](#get)
- [Examples](#examples)
- [Citing Us](#citing)

## News:
- 03 May 2020: v0.0.1 dEchorate is online

## Get the Data
The data is available at [Zenodo](www.notavailableyet.com).
Please, follow that link to download (part of) the data.

The dataset is available in multiple ways:
- Annotations/Labels: csv file that can be used pandas (Python)
- Only RIRs: a numpy matrix `n_samples x n_mics x n_srcs x n_rooms` (~ 6 GB)
- Only Speech: a numpy matrix `n_samples x n_mics x n_srcs x n_rooms` (~ 6 GB)
- Raw data: an hdf5 file (~ 75 GB) containig all the raw recording data (chirps, speech/noise sources, babble noise, room tone)

## Examples

## Citing dEchorate

If you are using `dEchorate` and you want to cite us, please use

```BibTex
@preprint{di2021dechorate,
    title={dEchorate: a Calibrated Room Impulse Response Database for Echo-aware Signal Processing},
    author={Di Carlo, Diego and Tandeitnik, Pinchas and Foy, C{\'e}dric and Deleforge, Antoine and Bertin, Nancy and Gannot, Sharon},
    journal={arXiv preprint arXiv:2104.13168},
    year={2021}
  }
```
