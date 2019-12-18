import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt

# h <- [pyroomacoustics, realRIR]
# x <- [chirp]
# windowing <- [on/off]
# snr <- -10:30 dB
# rt60 <- 0: 1 seconds

# metrics
# thr for allclose
# wasserstein-distance?!