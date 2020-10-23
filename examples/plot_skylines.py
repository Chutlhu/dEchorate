import os
import numpy as np


path_to_rirs = os.path.join('data', 'final', 'rir_matrix.npy')
rirs = np.load(path_to_rirs)

print(rirs.shape)
