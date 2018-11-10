"""Uses a classifier to decode PSD values.

Computes pvalues and saves them in a mat format with the decoding accuracies.

Author: Arthur Dehgan
"""
import sys
from time import time
from itertools import product

# from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from scipy.io import savemat, loadmat
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from utils import (
    StratifiedLeave2GroupsOut,
    elapsed_time,
    create_groups,
    prepare_data,
    classification,
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
PREFIX = "bootstrapped_perm_subsamp_"
SOLVER = "svd"  # 'svd' 'lsqr'

PREF_LIST = PREFIX.split("_")
BOOTSTRAP = "bootstrapped" in PREF_LIST
SUBSAMPLE = "subsamp" in PREF_LIST
PERM = "perm" in PREF_LIST
N_PERM = 999 if PERM else None
N_BOOTSTRAPS = 100 if BOOTSTRAP else 1

SAVE_PATH /= NAME


def classif_psd(state, elec):
    global SUBSAMPLE, SAVE_PATH
    if SUBSAMPLE:
        info_data = pd.read_csv(SAVE_PATH.parent / "info_data.csv")[STATE_LIST]
        N_TRIALS = info_data.min().min()
        N_SUBS = len(info_data) - 1
        groups = [i for i in range(N_SUBS) for _ in range(N_TRIALS)]
        N_TOTAL = N_TRIALS * N_SUBS
        labels = [0 if i < N_TOTAL / 2 else 1 for i in range(N_TOTAL)]
    else:
        labels = loadmat(LABEL_PATH / state + "_labels.mat")["y"].ravel()
        labels, groups = create_groups(labels)

    for freq in FREQ_DICT:
        print(state, elec, freq)

        data_file_name = NAME + "_{}_{}_{}_{}_{:.2f}.mat".format(
            state, freq, elec, WINDOW, OVERLAP
        )

        save_file_name = PREFIX + data_file_name

        data_file_path = SAVE_PATH / data_file_name

        save_file_path = SAVE_PATH / "results" / save_file_name

        if not save_file_path.isfile():
            n_rep = 0
        else:
            final_save = proper_loadmat(save_file_path)
            n_rep = int(final_save["n_rep"])
        print("Starting from i={}".format(n_rep))

        if not save_file_path.isfile():
            for i in range(n_rep, N_BOOTSTRAPS):
                data = loadmat(data_file_path)
                if SUBSAMPLE:
                    data = prepare_data(data, n_trials=N_TRIALS, random_state=i)
                else:
                    data = prepare_data(data)

                data = np.array(data).reshape(len(data), 1)
                sl2go = StratifiedLeave2GroupsOut()
                clf = LDA(solver=SOLVER)
                save = classification(
                    clf, sl2go, data, labels, groups, N_PERM, n_jobs=-1
                )

                print(save["acc_score"])
                if i == 0:
                    final_save = save
                elif BOOTSTRAP:
                    for key, value in save.items():
                        final_save[key] += value

                final_save["n_rep"] = i + 1
                savemat(file_path, final_save)

            final_save["n_rep"] = N_BOOTSTRAPS
            if BOOTSTRAP:
                final_save["auc_score"] = np.mean(final_save["auc_score"])
                final_save["acc_score"] = np.mean(final_save["acc_score"])
            savemat(save_file_path, final_save)

            print(
                "accuracy for %s %s : %0.2f (+/- %0.2f)"
                % (state, elec, final_save["acc_score"], np.std(final_save["acc"])
            )
            if PERM:
                print(
                    "{} : {:.2f} significatif a p={:.4f}".format(
                        freq, final_save["acc_score"], final_save["acc_pvalue"]
                    )
                )
            else:
                print("{} : {:.2f}".format(freq, save["acc_score"]))


if __name__ == "__main__":
    TIMELAPSE_START = time()
    ARGS = sys.argv[1:][0].split("_")
    if ARGS == []:
        for state, elec in product(STATE_LIST, CHANNEL_NAMES):
            classif_psd(state, elec)
    else:
        print(ARGS)
        classif_psd(ARGS[0], ARGS[1])
    print("total time lapsed : %s" % (elapsed_time(TIMELAPSE_START, time())))
