'''Uses a classifier to decode PSD values.

Computes pvalues and saves them in a mat format with the decoding accuracies.

Author: Arthur Dehgan
'''
from time import time
# from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from numpy.random import permutation
from scipy.io import savemat, loadmat
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score
from utils import StratifiedLeave2GroupsOut, elapsed_time, create_groups, prepare_data
from params import SAVE_PATH, LABEL_PATH, path, CHANNEL_NAMES,\
                   WINDOW, OVERLAP, STATE_LIST, FREQ_DICT

N_PERMUTATIONS = 999
SAVE_PATH = SAVE_PATH / 'psd'
SOLVER = 'svd'  # 'svd' 'lsqr'
SUBSAMPLE = True
PERM = False


def classification(state, elec):
    if SUBSAMPLE:
        info_data = pd.read_csv('info_data.csv')[STATE_LIST]
        N_TRIALS = info_data.min().min()
        N_SUBS = len(info_data) - 1
        groups = [i for _ in range(N_TRIALS) for i in range(N_SUBS)]
        N_TOTAL = N_TRIALS * N_SUBS
        labels = [0 if i < N_TOTAL / 2 else 1 for i in range(N_TOTAL)]
    else:
        labels = loadmat(LABEL_PATH / state + '_labels.mat')['y'].ravel()
        labels, groups = create_groups(labels)

    for key in FREQ_DICT:
        print(state, elec, key)

        file_name = 'classif_subsamp_PSD_{}_{}_{}_{}_{:.2f}.mat'.format(
                state, key, elec, WINDOW, OVERLAP)
        # file_name = 'perm_PSD_{}_{}_{}_{}_{:.2f}.mat'.format(
        #         state, key, elec, WINDOW, OVERLAP)
        results_file_path = SAVE_PATH / 'results/{}_solver'.format(SOLVER) / file_name

        if not path(results_file_path).isfile():
            # print('\nloading PSD for {} frequencies'.format(key))
            data_file_path = SAVE_PATH /\
                    'PSD_{}_{}_{}_{}_{:.2f}.mat'.format(
                        state, key, elec, WINDOW, OVERLAP)
            if path(data_file_path).isfile():
                data = prepare_data(loadmat(data_file_path))
            else:
                print(path(data_file_path).name + ' Not found')
                print('please run "computePSD.py" and\
                      "group_PSD_per_subjects.py" before\
                      running this script')

            # print('classification...')
            data = np.array(data).reshape(len(data), 1)
            sl2go = StratifiedLeave2GroupsOut()
            clf = LDA(solver=SOLVER)
            pvalue = 0
            good_score = cross_val_score(cv=sl2go,
                                         estimator=clf,
                                         X=data, y=labels,
                                         groups=groups,
                                         n_jobs=-1).mean()
            data = {'score': good_score}

            if PERM:
                pscores = []
                for _ in range(N_PERMUTATIONS):
                    clf = LDA()
                    perm_set = permutation(len(labels))
                    labels_perm = labels[perm_set]
                    groups_perm = groups[perm_set]
                    pscores.append(cross_val_score(cv=sl2go,
                                                  estimator=clf,
                                                  X=data, y=labels_perm,
                                                  groups=groups_perm,
                                                  n_jobs=-1).mean())

                for score in pscores:
                    if good_score <= score:
                        pvalue += 1/(N_PERMUTATIONS)

                data['pvalue'] = pvalue
                data['pscore'] = pscores

                print('{} : {:.2f} significatif a p={:.4f}'.format(
                    key, good_score, pvalue))

            else:
                print('{} : {:.2f}'.format(key, good_score))

            savemat(results_file_path, data)


if __name__ == '__main__':
    T0 = time()

    # Parallel(n_jobs=-1)(delayed(classification)(state,
    #                                             elec)
    #                     for state in STATE_LIST for elec in CHANNEL_NAMES)
    for state in STATE_LIST:
        for elec in CHANNEL_NAMES:
            classification(state, elec)

    print('total time lapsed : {}'.format(elapsed_time(T0, time())))
