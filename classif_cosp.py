"""Load crosspectrum matrix, perform classif, perm test, saves results.

Outputs one file per freq x state

Author: Arthur Dehgan"""
import sys
from time import time
from itertools import product
from scipy.io import savemat, loadmat
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedShuffleSplit as SSS
from pyriemann.classification import TSclassifier
from utils import (
    create_groups,
    StratifiedShuffleGroupSplit,
    elapsed_time,
    prepare_data,
    classification,
    proper_loadmat,
)
from params import SAVE_PATH, FREQ_DICT, STATE_LIST, WINDOW, OVERLAP, LABEL_PATH

# PREFIX = "perm_"
# PREFIX = "classif_"
# PREFIX = "reduced_classif_"
PREFIX = "bootstrapped_subsamp_"
NAME = "cosp"
# NAME = "cosp"
# NAME = 'ft_cosp'
# NAME = "moy_cosp"
# NAME = 'im_cosp'
# NAME = 'wpli'
# NAME = 'coh'
# NAME = 'imcoh'
# NAME = 'ft_wpli'
# NAME = 'ft_coh'
# NAME = 'ft_imcoh'

PREFIX_LIST = PREFIX.split("_")
BOOTSTRAP = "bootstrapped" in PREFIX_LIST
SUBSAMPLE = "subsamp" in PREFIX_LIST
ADAPT = "adapt" in PREFIX_LIST
PERM = "perm" in PREFIX_LIST
FULL_TRIAL = "ft" in NAME or "moy" in NAME.split("_")
N_PERM = 999 if PERM else None
N_BOOTSTRAPS = 1 if BOOTSTRAP else 1

SAVE_PATH /= NAME


def classif_cosp(state, freq, n_jobs=-1):
    print(state, freq)
    if FULL_TRIAL:
        labels = np.concatenate((np.ones(18), np.zeros(18)))
        groups = range(36)
    elif SUBSAMPLE:
        info_data = pd.read_csv(SAVE_PATH.parent / "info_data.csv")[STATE_LIST]
        n_trials = info_data.min().min()
        n_subs = len(info_data) - 1
        groups = [i for i in range(n_subs) for _ in range(n_trials)]
        n_total = n_trials * n_subs
        labels = [0 if i < n_total / 2 else 1 for i in range(n_total)]
    elif ADAPT:
        info_data = pd.read_csv(SAVE_PATH.parent / "info_data.csv")[STATE_LIST]
        n_trials = info_data.min()[state]
        n_subs = len(info_data) - 1
        groups = [i for i in range(n_subs) for _ in range(n_trials)]
        n_total = n_trials * n_subs
        labels = [0 if i < n_total / 2 else 1 for i in range(n_total)]
    else:
        labels = loadmat(LABEL_PATH / state + "_labels.mat")["y"].ravel()
        labels, groups = create_groups(labels)

    file_path = (
        SAVE_PATH / "results" / PREFIX
        + NAME
        + "_{}_{}_{}_{:.2f}.mat".format(state, freq, WINDOW, OVERLAP)
    )

    if not file_path.isfile():
        n_rep = 0
    else:
        final_save = proper_loadmat(file_path)
        n_rep = int(final_save["n_rep"])
    print("Starting from i={}".format(n_rep))

    file_name = NAME + "_{}_{}_{}_{:.2f}.mat".format(state, freq, WINDOW, OVERLAP)
    data_file_path = SAVE_PATH / file_name

    data_og = loadmat(data_file_path)
    for i in range(n_rep, N_BOOTSTRAPS):
        if FULL_TRIAL:
            data = data_og["data"]
        elif SUBSAMPLE or ADAPT:
            data = prepare_data(data_og, n_trials=n_trials, random_state=i)
        else:
            data = prepare_data(data_og)

        if FULL_TRIAL:
            crossval = SSS(9)
        else:
            crossval = StratifiedShuffleGroupSplit(2)
        lda = LDA()
        clf = TSclassifier(clf=lda)
        save = classification(
            clf, crossval, data, labels, groups, N_PERM, n_jobs=n_jobs
        )

        print(save["acc_score"])
        if i == 0:
            final_save = save
        elif BOOTSTRAP:
            for key, value in save.items():
                final_save[key] += value

        final_save["n_rep"] = i + 1
        savemat(file_path, final_save)

    if BOOTSTRAP:
        final_save["auc_score"] = np.mean(final_save["auc_score"])
        final_save["acc_score"] = np.mean(final_save["acc_score"])
    savemat(file_path, final_save)

    print(
        "accuracy for %s %s : %0.2f (+/- %0.2f)"
        % (state, freq, np.mean(save["acc_score"]), np.std(save["acc"]))
    )
    if PERM:
        print("pval = {}".format(save["acc_pvalue"]))


if __name__ == "__main__":
    TIMELAPSE_START = time()
    ARGS = sys.argv
    if len(ARGS) > 2:
        ARGS = sys.argv[1:]
    elif len(ARGS) == 2:
        ARGS = sys.argv[1:][0].split("_")
    else:
        ARGS = []

    if ARGS == []:
        from joblib import delayed, Parallel

        Parallel(n_jobs=-1)(
            delayed(classif_cosp)(st, fr, n_jobs=1)
            for st, fr in product(STATE_LIST, FREQ_DICT)
        )
    else:
        print(ARGS)
        classif_cosp(ARGS[0], ARGS[1])
    print("total time lapsed : %s" % (elapsed_time(TIMELAPSE_START, time())))
