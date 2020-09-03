from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import permutations
from scipy.signal import find_peaks

from dechorate import constants
from dechorate.utils.file_utils import *

from mir_eval.onset import evaluate
from mir_eval.util import match_events


###############################################################################
##  ESTIMATION WITH CROCCO
#

curr_dir = './recipes/acoustic_echo_retrieval/'

exp = 'xval' # 'synt'

dataset_dir = './data/dECHORATE/'
path_to_processed = './data/processed/'
path_to_raw_zips = dataset_dir + '/recordings/room-%s.zip'
path_to_note_csv = dataset_dir + 'annotations/dECHORATE_database.csv'
path_to_after_calibration = path_to_processed + 'post2_calibration/calib_output_mics_srcs_pos.pkl'

path_to_interim = curr_dir + 'data/interim/' + exp + '/'
path_to_results = curr_dir + 'results/' + exp + '/'

K = 4
n_mics = [2,4,6]
targets = [0, 1, 2, 3]
datasets = [(2, '011000'), (4, '011110'), (5, '011111')]
data = ['synt', 'real']
thrs_smpl = [5]
lambdas = [0.001, 0.005, 0.01, 0.02, 0.05]

results = pd.DataFrame()


def eval_aer(path_to_results, path_to_data, thr_smpl):

    reference_toas = load_from_matlab(path_to_results)['toas_est'][0]
    print(reference_toas)
    print(reference_toas[0])
    reference_toas = np.concatenate(reference_toas, axis=1).reshape([K, 2])

    estimated_toas = load_from_matlab(path_to_results)['toas_ref'][0]
    print(estimated_toas)
    estimated_toas = np.concatenate(estimated_toas, axis=1).reshape([K, 2])



    # remove global delay
    reference_toas = reference_toas - np.min(reference_toas)
    estimated_toas = estimated_toas - np.min(estimated_toas)

    # search for the best permutation of channel
    perms = []
    precisions_perm = []
    for perm in permutations(range(M)):
        precisions_chan = []
        for m in range(M):

            ref = reference_toas[:, m]
            est = estimated_toas[:, perm[m]]

            tmp_scores = evaluate(ref, est, window=thr_smpl)
            precisions_chan.append(tmp_scores['Precision'])

        precisions_perm.append(np.sum(precisions_chan))
        perms.append(perm)

    best_perm = perms[np.argmax(precisions_perm)]
    best_precision = precisions_perm[np.argmax(precisions_perm)]

    # print results for the best match
    mean_f = []
    mean_p = []
    mean_r = []
    mean_e = []

    for m in range(M):

        ref = reference_toas[:, m]
        est = estimated_toas[:, best_perm[m]]

        # F, P, M
        chan_scores = evaluate(ref, est, window=thr_smpl)
        # RMSE
        mean_f.append(chan_scores['F-measure'])
        mean_p.append(chan_scores['Precision'])
        mean_r.append(chan_scores['Recall'])

        matching = np.array(match_events(ref, est, window=thr_smpl))
        if len(matching) == 0:
            rmse = np.nan
        else:
            rmse = np.sqrt(np.mean((ref[matching[:,0]] - est[matching[:,1]])**2))

        mean_e.append(rmse)

    mean_f = np.nanmean(mean_f)
    mean_p = np.nanmean(mean_p)
    mean_r = np.nanmean(mean_r)
    mean_e = np.nanmean(mean_e)

    res = {
        'mean_f' : mean_f,
        'mean_p' : mean_p,
        'mean_r' : mean_r,
        'mean_e' : mean_e,
    }

    return res

c = 0
for D, dset in tqdm(datasets):
    for t in targets:
        for M in n_mics:
            for d in data:
                for thr_smpl in thrs_smpl:
                    for l, lam in enumerate(lambdas):

                        mat_file_data = 'data4crocco_dataset-%d_target-%d.mat' % (D, t)
                        mat_file_res = 'data4crocco_dataset-%d_target-%d_nmic-%d_%s_multip-%d.mat' % (D, t, M, d, l)

                        path_to_math_file_data = path_to_interim + mat_file_data
                        path_to_math_file_res = path_to_interim + mat_file_res
                        try:
                            res = eval_aer(path_to_math_file_res, path_to_math_file_data, thr_smpl)
                        except FileNotFoundError:
                            continue

                        results.at[c, 'dataset'] = dset
                        results.at[c, 'data'] = d
                        results.at[c, 'n_mics'] = M
                        results.at[c, 'target'] = t
                        results.at[c, 'thr_smpl'] = thr_smpl
                        results.at[c, 'lambda'] = lam
                        results.at[c, 'mean_f'] = res['mean_f']
                        results.at[c, 'mean_p'] = res['mean_p']
                        results.at[c, 'mean_r'] = res['mean_r']
                        results.at[c, 'mean_e'] = res['mean_e']
                        c += 1

print(results)
results.to_csv(path_to_results + 'aer_crocco.csv')
