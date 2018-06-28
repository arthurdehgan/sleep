'''Uses a classifier to decode PSD values.

Computes pvalues and saves them in a mat format with the decoding accuracies.

Author: Arthur Dehgan
'''
from time import time
# from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from itertools import product
from scipy.io import savemat, loadmat
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from utils import StratifiedLeave2GroupsOut, elapsed_time, create_groups,\
                  prepare_data, classification
from params import SAVE_PATH, LABEL_PATH, path, CHANNEL_NAMES,\
                   WINDOW, OVERLAP, STATE_LIST, FREQ_DICT

# prefix = 'perm'
# prefix = 'classif'
prefix = 'classif_subsamp'
SAVE_PATH = SAVE_PATH / 'psd'
SOLVER = 'svd'  # 'svd' 'lsqr'
SUBSAMPLE = prefix.endswith('subsamp')
PERM = prefix.startswith('perm')
if PERM:
    N_PERM = 999
else:
    N_PERM = None
N_TRIALS = None


def main(state, elec):
    global N_TRIALS, SUBSAMPLE, SAVE_PATH
    if SUBSAMPLE:
        info_data = pd.read_csv('info_data.csv')[STATE_LIST]
        if N_TRIALS is None:
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

        file_name = prefix + '_PSD_{}_{}_{}_{}_{:.2f}.mat'.format(
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
                data = prepare_data(loadmat(data_file_path), n_trials=N_TRIALS)
            else:
                print(path(data_file_path).name + ' Not found')
                print('please run "computePSD.py" and\
                      "group_PSD_per_subjects.py" before\
                      running this script')

            # print('classification...')
            data = np.array(data).reshape(len(data), 1)
            sl2go = StratifiedLeave2GroupsOut()
            clf = LDA(solver=SOLVER)
            save = classification(clf, sl2go, data, labels, groups,
                                  N_PERM, n_jobs=-1)

            if PERM:
                print('{} : {:.2f} significatif a p={:.4f}'.format(
                    key, save['acc_score'], save['acc_pvalue']))
            else:
                print('{} : {:.2f}'.format(key, save['acc_score']))

            savemat(results_file_path, save)


if __name__ == '__main__':
    T0 = time()

    for state, elec in product(STATE_LIST, CHANNEL_NAMES):
            main(state, elec)

    print('total time lapsed : {}'.format(elapsed_time(T0, time())))
