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
thrs_smpl = [1, 3, 5]
lambdas = [0.001, 0.005, 0.01, 0.02, 0.05]

results = pd.DataFrame()


def eval_aer(path_to_results, path_to_data, thr_smpl):

    estimated_toas = load_from_matlab(path_to_results)['toas_est'][0]

    # compute the ground truth

    if 'real' in path_to_results:
        rirs = load_from_matlab(path_to_data)['rirs_real']
    elif 'synt' in path_to_results:
        rirs = load_from_matlab(path_to_data)['rirs_synt']
    else:
        raise ValueError('Not right rirs')

    M = len(estimated_toas)

    reference_toas = []
    for i in range(M):
        x = np.abs(rirs[:, i, 0])**2
        # find peaks in the RIRs
        peaks, _ = find_peaks(x)
        amps = x[peaks]

        sort_amps_index = np.argsort(amps)[::-1][:K]
        taus = peaks[sort_amps_index]
        amps = amps[sort_amps_index]

        sort_taus_index = np.argsort(taus)
        taus = taus[sort_taus_index]

        reference_toas.append(taus)

    # remove global delay
    reference_toas_min = np.min([np.min(x) for x in reference_toas])
    estimated_toas_min = np.min([np.min(x) for x in estimated_toas])

    print(reference_toas)
    print(estimated_toas)
    1/0

    # search for the best permutation of channel
    perms = []
    fmeasures_perm = []
    for perm in permutations(range(M)):
        fmeasures_chan = []
        for m in range(M):
            ref = np.array(reference_toas[perm[m]]).reshape([K]) - reference_toas_min

            est = np.array(estimated_toas[m]) - estimated_toas_min
            est = est.reshape([np.size(est)])
            if len(est) > 1:
                est = np.sort(est)

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
        ref = np.array(reference_toas[perm[m]]).reshape([K]) - reference_toas_min
        ref = np.sort(ref)

        est = np.array(estimated_toas[m]) - estimated_toas_min
        est = est.reshape([np.size(est)])
        if len(est) > 1:
            est = np.sort(est)

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
