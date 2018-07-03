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
prefix = 'classif_subsamp_'
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
FULL_TRIAL = name.startswith('ft') or name.startswith('moy')
SUBSAMPLE = prefix.endswith('subsamp_')
PERM = prefix.startswith('perm')
if PERM:
    N_PERM = 999
else:
    N_PERM = None
SAVE_PATH = SAVE_PATH / name


def main(state, freq):
    """Where the magic happens"""
    print(state, freq)
    if FULL_TRIAL:
        labels = np.concatenate((np.ones(18,), np.zeros(18,)))
        groups = range(36)
    elif SUBSAMPLE:
        info_data = pd.read_csv('info_data.csv')[STATE_LIST]
        N_TRIALS = info_data.min().min()
        N_SUBS = len(info_data) - 1
        groups = [i for _ in range(N_TRIALS) for i in range(N_SUBS)]
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
            data = loadmat(data_file_path)

            if FULL_TRIAL:
                data = data['data']
            else:
                data = prepare_data(data)

            sl2go = StratifiedLeave2GroupsOut()
            lda = LDA()
            clf = TSclassifier(clf=lda)
            save = classification(clf, sl2go, data, labels,
                                  groups, N_PERM, n_jobs=-1)

            savemat(file_path, save)

            print('accuracy for %s %s : %0.2f (+/- %0.2f)' %
                  (state, freq, save['acc_score'], np.std(save['acc'])))

        else:
            print(data_file_path.name + ' Not found')


if __name__ == '__main__':
    TIMELAPSE_START = time()
    for freq, state in product(FREQ_DICT, STATE_LIST):
            main(state, freq)
    print('total time lapsed : %s' % elapsed_time(TIMELAPSE_START, time()))
