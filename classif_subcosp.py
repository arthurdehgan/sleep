"""Load crosspectrum matrix, perform classif, perm test, saves results.

Outputs one file per freq x state

Author: Arthur Dehgan"""
import sys
from time import time
from itertools import product
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedShuffleSplit as SSS
from pyriemann.classification import TSclassifier
from utils import (
    StratifiedShuffleGroupSplit,
    elapsed_time,
    prepare_data,
    classification,
    proper_loadmat,
)
from params import SAVE_PATH, FREQ_DICT, STATE_LIST, WINDOW, OVERLAP, CHANNEL_NAMES

STATE_LIST = ["S2", "SWS"]

PREFIX = "bootstrapped_subsamp_"
NAME = "cosp"

PREFIX_LIST = PREFIX.split("_")
BOOTSTRAP = "bootstrapped" in PREFIX_LIST
SUBSAMPLE = "subsamp" in PREFIX_LIST
ADAPT = "adapt" in PREFIX_LIST
PERM = "perm" in PREFIX_LIST
FULL_TRIAL = "ft" in NAME or "moy" in NAME.split("_")
N_PERM = 999 if PERM else None
N_BOOTSTRAPS = 1000 if BOOTSTRAP else 1
INIT_LABELS = [0] * 18 + [1] * 18
CHANGES = False

SAVE_PATH /= NAME


def classif_subcosp(state, freq, elec, n_jobs=-1):
    global CHANGES
    print(state, freq)
    if SUBSAMPLE or ADAPT:
        info_data = pd.read_csv(SAVE_PATH.parent / "info_data.csv")[STATE_LIST]
        if SUBSAMPLE:
            n_trials = info_data.min().min()
            n_trials = 61
        elif ADAPT:
            n_trials = info_data.min()[state]
    elif FULL_TRIAL:
        groups = range(36)
    labels_og = INIT_LABELS

    file_path = (
        SAVE_PATH / "results" / PREFIX
        + NAME
        + "_{}_{}_{}_{}_{:.2f}.npy".format(state, freq, elec, WINDOW, OVERLAP)
    )

    if not file_path.isfile():
        n_rep = 0
    else:
        final_save = np.load(file_path)
        n_rep = int(final_save["n_rep"])
        n_splits = int(final_save["n_splits"])
    print("Starting from i={}".format(n_rep))

    file_name = NAME + "_{}_{}_{}_{}_{:.2f}.npy".format(
        state, freq, elec, WINDOW, OVERLAP
    )
    data_file_path = SAVE_PATH / file_name

    data_og = np.load(data_file_path)
    if FULL_TRIAL:
        cv = SSS(9)
    else:
        cv = StratifiedShuffleGroupSplit(2)
    lda = LDA()
    clf = TSclassifier(clf=lda)

    for i in range(n_rep, N_BOOTSTRAPS):
        CHANGES = True
        if FULL_TRIAL:
            data = data_og["data"]
        elif SUBSAMPLE or ADAPT:
            data, labels, groups = prepare_data(
                data_og, labels_og, n_trials=n_trials, random_state=i
            )
        else:
            data, labels, groups = prepare_data(data_og, labels_og)
        n_splits = cv.get_n_splits(None, labels, groups)

        save = classification(clf, cv, data, labels, groups, N_PERM, n_jobs=n_jobs)

        if i == 0:
            final_save = save
        elif BOOTSTRAP:
            for key, value in save.items():
                if key != "n_splits":
                    final_save[key] += value

        final_save["n_rep"] = i + 1
        np.save(file_path, final_save)

    final_save["auc_score"] = np.mean(final_save.get("auc_score", 0))
    final_save["acc_score"] = np.mean(final_save["acc_score"])
    if CHANGES:
        np.save(file_path, final_save)

    to_print = "accuracy for {} {} : {:.2f}".format(
        state, freq, final_save["acc_score"]
    )
    if BOOTSTRAP:
        standev = np.std(
            [
                np.mean(final_save["acc"][i * n_splits : (i + 1) * n_splits])
                for i in range(N_BOOTSTRAPS)
            ]
        )
        to_print += " (+/- {:.2f})".format(standev)
    print(to_print)
    if PERM:
        print("pval = {}".format(final_save["acc_pvalue"]))


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

        Parallel(n_jobs=1)(
            delayed(classif_subcosp)(st, fr, el, n_jobs=1)
            for st, fr, el in product(STATE_LIST, FREQ_DICT, CHANNEL_NAMES)
        )
    else:
        print(ARGS)
        classif_subcosp(ARGS[0], ARGS[1], ARGS[2])
    print("total time lapsed : %s" % (elapsed_time(TIMELAPSE_START, time())))
