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
prefix = 'bootstrapped_perm_subsamp'
SOLVER = 'svd'  # 'svd' 'lsqr'

pref_list = prefix.split('_')
BOOTSTRAP = 'bootstrapped' in pref_list
SUBSAMPLE = 'subsamp' in pref_list
PERM = 'perm' in pref_list
N_PERM = 99 if PERM else None
N_BOOTSTRAPS = 10 if BOOTSTRAP else None

SAVE_PATH = SAVE_PATH / 'psd'


def main(state, elec):
    global N_TRIALS, SUBSAMPLE, SAVE_PATH
    if SUBSAMPLE:
        info_data = pd.read_csv(SAVE_PATH.parent / 'info_data.csv')[STATE_LIST]
        N_TRIALS = info_data.min().min()
        N_SUBS = len(info_data) - 1
        groups = [i for _ in range(N_TRIALS) for i in range(N_SUBS)]
        N_TOTAL = N_TRIALS * N_SUBS
        labels = [0 if i < N_TOTAL / 2 else 1 for i in range(N_TOTAL)]
    else:
        labels = loadmat(LABEL_PATH / state + '_labels.mat')['y'].ravel()
        labels, groups = create_groups(labels)

    for freq in FREQ_DICT:
        print(state, elec, freq)

        file_name = prefix + '_PSD_{}_{}_{}_{}_{:.2f}.mat'.format(
                state, freq, elec, WINDOW, OVERLAP)

        save_file_path = SAVE_PATH / 'results' / file_name

        if not save_file_path.isfile():
            data_file_path = SAVE_PATH / 'PSD_{}_{}_{}_{}_{:.2f}.mat'.format(
                        state, freq, elec, WINDOW, OVERLAP)

            if path(data_file_path).isfile():
                final_save = None

                for i in range(N_BOOTSTRAPS):
                    data = loadmat(data_file_path)
                    if SUBSAMPLE:
                        data = prepare_data(data,
                                            n_trials=N_TRIALS,
                                            random_state=i)
                    else:
                        data = prepare_data(data)

                    data = np.array(data).reshape(len(data), 1)
                    sl2go = StratifiedLeave2GroupsOut()
                    clf = LDA(solver=SOLVER)
                    save = classification(clf, sl2go, data, labels, groups,
                                          N_PERM, n_jobs=-1)
                    save['acc_bootstrap'] = [save['acc_score']]
                    save['auc_bootstrap'] = [save['auc_score']]
                    if final_save is None:
                        final_save = save
                    else:
                        for key, value in final_save.items():
                            final_save[key] = final_save[key] + save[key]

                savemat(save_file_path, final_save)

                if PERM:
                    print('{} : {:.2f} significatif a p={:.4f}'.format(
                        freq, save['acc_score'], save['acc_pvalue']))
                else:
                    print('{} : {:.2f}'.format(freq, save['acc_score']))

            else:
                print(data_file_path.name + ' Not found')


if __name__ == '__main__':
    TIMELAPSE_START = time()
    for state, elec in product(STATE_LIST, CHANNEL_NAMES):
            main(state, elec)
    print('total time lapsed : %s' % (elapsed_time(TIMELAPSE_START, time())))
