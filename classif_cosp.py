"""Load crosspectrum matrix, perform classif, perm test, saves results.

Outputs one file per freq x state

Author: Arthur Dehgan"""
from time import time
from itertools import product
from scipy.io import savemat, loadmat
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from pyriemann.classification import TSclassifier
from utils import create_groups, StratifiedLeave2GroupsOut, elapsed_time,\
    prepare_data, classification
from params import SAVE_PATH, FREQ_DICT, STATE_LIST, WINDOW,\
    OVERLAP, LABEL_PATH
# import pdb

# prefix = 'perm_'
# prefix = 'classif_'
# prefix = 'classif_reduced'
prefix = 'bootstrapped_classif_subsamp_'
name = 'cosp'
# name = 'ft_cosp'
# name = 'moy_cosp'
# name = 'im_cosp'
# name = 'wpli'
# name = 'coh'
# name = 'imcoh'
# name = 'ft_wpli'
# name = 'ft_coh'
# name = 'ft_imcoh'
pref_list = prefix.split('_')
if len(pref_list) > 1:
    save_prefix = pref_list[-1]
BOOTSTRAP = 'bootstrapped' in pref_list
REDUCED = 'reduced' in pref_list
FULL_TRIAL = 'ft' in pref_list or 'moy' in pref_list
SUBSAMPLE = 'subsamp' in pref_list
PERM = 'perm' in pref_list
N_PERM = 999 if PERM else None
N_BOOTSTRAPS = 10 if BOOTSTRAP else 1
N_BOOTSTRAPS = 19 if REDUCED else 1

SAVE_PATH = SAVE_PATH / name
print(name, prefix)


def main(state, freq):
    """Where the magic happens"""
    print(state, freq)
    if FULL_TRIAL:
        labels = np.concatenate((np.ones(18,), np.zeros(18,)))
        groups = range(36)
    elif SUBSAMPLE:
        info_data = pd.read_csv(SAVE_PATH.parent / 'info_data.csv')[STATE_LIST]
        N_TRIALS = info_data.min().min()
        N_SUBS = len(info_data) - 1
        groups = [i for i in range(N_SUBS) for _ in range(N_TRIALS)]
        N_TOTAL = N_TRIALS * N_SUBS
        labels = [0 if i < N_TOTAL / 2 else 1 for i in range(N_TOTAL)]
    else:
        labels = loadmat(LABEL_PATH / state + '_labels.mat')['y'].ravel()
        labels, groups = create_groups(labels)

    file_path = SAVE_PATH / 'results' /\
        prefix + name + '_{}_{}_{}_{:.2f}.mat'.format(
            state, freq, WINDOW, OVERLAP)

    if not file_path.isfile():
        file_name = name + '_{}_{}_{}_{:.2f}.mat'.format(
            state, freq, WINDOW, OVERLAP)
        data_file_path = SAVE_PATH / file_name

        if data_file_path.isfile():
            final_save = None

            for i in range(N_BOOTSTRAPS):
                data = loadmat(data_file_path)
                if FULL_TRIAL:
                    data = data['data']
                elif SUBSAMPLE:
                    data = prepare_data(data,
                                        n_trials=N_TRIALS,
                                        random_state=i)
                else:
                    data = prepare_data(data)

                if REDUCED:
                    reduced_data = []
                    for submat in data:
                        b = np.delete(submat, i, 0)
                        c = np.delete(b, i, 1)
                        reduced_data.append(c)
                    data = np.asarray(reduced_data)

                sl2go = StratifiedLeave2GroupsOut()
                lda = LDA()
                clf = TSclassifier(clf=lda)
                save = classification(clf, sl2go, data, labels,
                                      groups, N_PERM, n_jobs=-1)

                if BOOTSTRAP or REDUCED:
                    if i == 0:
                        save['acc_' + save_prefix] = [save['acc_score']]
                        save['auc_' + save_prefix] = [save['auc_score']]
                    else:
                        save['acc_' + save_prefix] += save['acc_score']
                        save['auc_' + save_prefix] += save['auc_score']

            savemat(file_path, final_save)

            print('accuracy for %s %s : %0.2f (+/- %0.2f)' %
                  (state, freq, save['acc_score'], np.std(save['acc'])))

        else:
            print(data_file_path.name + ' Not found')


if __name__ == '__main__':
    TIMELAPSE_START = time()
    for freq, state in product(FREQ_DICT, STATE_LIST):
            main(state, freq)
    print('total time lapsed : %s' % elapsed_time(TIMELAPSE_START, time()))
