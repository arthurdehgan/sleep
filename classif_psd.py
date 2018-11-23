"""Uses a classifier to decode PSD values.

Computes pvalues and saves them in a mat format with the decoding accuracies.

Author: Arthur Dehgan
"""
import sys
from time import time
from itertools import product
import pandas as pd
import numpy as np
from scipy.io import savemat, loadmat
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from utils import (
    StratifiedShuffleGroupSplit,
    elapsed_time,
    create_groups,
    prepare_data,
    classification,
    proper_loadmat,
)
from params import (
    SAVE_PATH,
    LABEL_PATH,
    CHANNEL_NAMES,
    WINDOW,
    OVERLAP,
    STATE_LIST,
    FREQ_DICT,
)

NAME = "psd"
# NAME = "zscore_psd"
# PREFIX = "perm_"
# PREFIX = "bootstrapped_perm_subsamp_"
PREFIX = "bootstrapped_subsamp_"
# PREFIX = "bootstrapped_adapt_"
SOLVER = "svd"  # 'svd' 'lsqr'

PREF_LIST = PREFIX.split("_")
BOOTSTRAP = "bootstrapped" in PREF_LIST
SUBSAMPLE = "subsamp" in PREF_LIST
ADAPT = "adapt" in PREF_LIST
PERM = "perm" in PREF_LIST
N_PERM = 999 if PERM else None
N_BOOTSTRAPS = 1000 if BOOTSTRAP else 1
INIT_LABELS = [0] * 18 + [1] * 18

SAVE_PATH /= NAME


def classif_psd(state, elec, n_jobs=-1):
    if SUBSAMPLE or ADAPT:
        info_data = pd.read_csv(SAVE_PATH.parent / "info_data.csv")[STATE_LIST]
        labels = INIT_LABELS
        if SUBSAMPLE:
            n_trials = info_data.min().min()
            # n_trials = 30
        elif ADAPT:
            n_trials = info_data.min()[state]

    for freq in FREQ_DICT:
        print(state, elec, freq)
        data_file_name = NAME + "_{}_{}_{}_{}_{:.2f}.mat".format(
            state, freq, elec, WINDOW, OVERLAP
        )

        save_file_name = PREFIX + str(n_trials) + "_" + data_file_name

        data_file_path = SAVE_PATH / data_file_name

        save_file_path = SAVE_PATH / "results" / save_file_name

        if not save_file_path.isfile():
            n_rep = 0
        else:
            final_save = proper_loadmat(save_file_path)
            n_rep = int(final_save["n_rep"])
        print("Starting from i={}".format(n_rep))

        data = loadmat(data_file_path)
        crossval = StratifiedShuffleGroupSplit(4, 400)
        clf = LDA(solver=SOLVER)

        for i in range(n_rep, N_BOOTSTRAPS):
            if SUBSAMPLE or ADAPT:
                data, labels, groups = prepare_data(
                    data, labels, rm_outl=2, n_trials=n_trials, random_state=i
                )
            else:
                data, labels, groups = prepare_data(data, labels)
            n_splits = crossval.get_n_splits(None, labels, groups)

            data = np.array(data).reshape(len(data), 1)
            save = classification(
                clf, crossval, data, labels, groups, N_PERM, n_jobs=n_jobs
            )

            if i == 0:
                final_save = save
            elif BOOTSTRAP:
                for key, value in save.items():
                    final_save[key] += value

            final_save["n_rep"] = i + 1

            if n_jobs == -1:
                savemat(save_file_path, final_save)

        if BOOTSTRAP:
            final_save["auc_score"] = np.mean(final_save["auc_score"])
            final_save["acc_score"] = np.mean(final_save["acc_score"])
        savemat(save_file_path, final_save)

        standev = np.std(
            [
                np.mean(final_save["acc"][i * n_splits : (i + 1) * n_splits])
                for i in range(N_BOOTSTRAPS)
            ]
        )
        print(
            "accuracy for {} {} : {:.2f} (+/- {:.2f})".format(
                state, elec, final_save["acc_score"], standev
            )
        )
        if PERM:
            print("pval = {:.4f}".format(final_save["acc_pvalue"]))


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
            delayed(classif_psd)(st, el, n_jobs=1)
            for st, el in product(STATE_LIST, CHANNEL_NAMES)
        )
    else:
        print(ARGS)
        classif_psd(ARGS[0], ARGS[1])
    print("total time lapsed : %s" % (elapsed_time(TIMELAPSE_START, time())))
