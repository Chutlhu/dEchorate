from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import permutations

from dechorate import constants
from dechorate.utils.file_utils import *

from mir_eval.onset import evaluate
from mir_eval.util import match_events


###############################################################################
##  ESTIMATION WITH CROCCO
#

curr_dir = './recipes/acoustic_echo_retrieval/'

exp = 'synt'

dataset_dir = './data/dECHORATE/'
path_to_processed = './data/processed/'
path_to_raw_zips = dataset_dir + '/recordings/room-%s.zip'
path_to_note_csv = dataset_dir + 'annotations/dECHORATE_database.csv'
path_to_after_calibration = path_to_processed + 'post2_calibration/calib_output_mics_srcs_pos.pkl'

path_to_interim = curr_dir + 'data/interim/' + exp + '/'
path_to_results = curr_dir + 'results/' + exp + '/'


n_mics = [2,4,6]
targets = [0, 1, 2, 3]
datasets = [(2, '011000'), (4, '011110'), (5, '011111')]
data = ['synt', 'real']
thrs_smpl = [1, 3, 5]

results = pd.DataFrame()


def eval_aer(path_to_file, thr_smpl):

    # print(path_to_file)
    estimated_toas = load_from_matlab(path_to_file)['toas_est'][0]
    reference_toas = load_from_matlab(path_to_file)['toas_ref'][0]

    # remove global delay
    reference_toas_min = np.min([np.min(x) for x in reference_toas])
    estimated_toas_min = np.min([np.min(x) for x in estimated_toas])

    M = reference_toas.shape[0]

    # search for the best permutation of channel
    perms = []
    fmeasures_perm = []
    for perm in permutations(range(M)):
        fmeasures_chan = []
        for m in range(M):
            ref = np.sort(np.array(reference_toas[perm[m]]).squeeze() - reference_toas_min)
            est = np.sort(np.array(estimated_toas[m]).squeeze() - estimated_toas_min)
            tmp_scores = evaluate(ref, est, window=thr_smpl)
            fmeasures_chan.append(tmp_scores['F-measure'])

        fmeasures_perm.append(np.sum(fmeasures_chan))
        perms.append(perm)

    best_perm = perms[np.argmax(fmeasures_perm)]
    fmeasure = fmeasures_perm[np.argmax(fmeasures_perm)]

    # print results for the best match
    mean_f = []
    mean_p = []
    mean_r = []
    mean_e = []

    for m in range(M):
        ref = np.sort(np.array(reference_toas[best_perm[m]]).squeeze() - reference_toas_min)
        est = np.sort(np.array(estimated_toas[m]).squeeze() - estimated_toas_min)
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

                    mat_file = 'data4crocco_dataset-%d_target-%d_nmic-%d_%s' % (D, t, M, d)
                    path_to_math_file = path_to_interim + mat_file
                    try:
                        res = eval_aer(path_to_math_file, thr_smpl)
                    except FileNotFoundError:
                        continue


                    results.at[c, 'dataset'] = dset
                    results.at[c, 'data'] = d
                    results.at[c, 'n_mics'] = M
                    results.at[c, 'target'] = t
                    results.at[c, 'thr_smpl'] = thr_smpl
                    results.at[c, 'mean_f'] = res['mean_f']
                    results.at[c, 'mean_p'] = res['mean_p']
                    results.at[c, 'mean_r'] = res['mean_r']
                    results.at[c, 'mean_e'] = res['mean_e']
                    c += 1

print(results)
results.to_csv(path_to_results + 'aer_crocco.csv')
