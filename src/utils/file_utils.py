import pickle as pkl
from scipy.io import loadmat

def save_to_pickle(filename, obj):
    with open(filename, 'wb') as handle:
        pkl.dump(obj, handle, protocol=pkl.HIGHEST_PROTOCOL)


def load_from_pickle(filename):
    with open('filename', 'rb') as handle:
        b = pkl.load(handle)
    return b

def load_from_mat(filename):
    return loadmat(filename)