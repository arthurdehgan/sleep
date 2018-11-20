"""Uses a classifier to decode PSD values.

Computes pvalues and saves them in a mat format with the decoding accuracies.

Author: Arthur Dehgan
"""
from itertools import product
import pandas as pd
import numpy as np
from scipy.io import savemat, loadmat
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from joblib import delayed, Parallel
from utils import (
    StratifiedLeave2GroupsOut,
    prepare_data,
    classification,
    proper_loadmat,
)
from params import SAVE_PATH, CHANNEL_NAMES, WINDOW, OVERLAP, STATE_LIST, FREQ_DICT

NAME = "psd"
PREFIX = "perm_subsamp_"
SOLVER = "svd"  # 'svd' 'lsqr'

PREF_LIST = PREFIX.split("_")
SUBSAMPLE = "subsamp" in PREF_LIST
PERM = "perm" in PREF_LIST
N_PERM = 999 if PERM else None

SAVE_PATH /= NAME


def classif_psd(state, elec, freq, n_jobs=-1):
    info_data = pd.read_csv(SAVE_PATH.parent / "info_data.csv")[STATE_LIST]
    n_trials = info_data.min().min()
    n_subs = len(info_data) - 1
    groups = [i for i in range(n_subs) for _ in range(n_trials)]
    n_total = n_trials * n_subs
    labels = [0 if i < n_total / 2 else 1 for i in range(n_total)]

    print(state, elec, freq)
    data_file_name = NAME + "_{}_{}_{}_{}_{:.2f}.mat".format(
        state, freq, elec, WINDOW, OVERLAP
    )

    save_file_name = PREFIX + data_file_name

    data_file_path = SAVE_PATH / data_file_name

    save_file_path = SAVE_PATH / "results" / save_file_name

    data = loadmat(data_file_path)
    data = prepare_data(data, n_trials=n_trials, random_state=666)

    data = np.array(data).reshape(len(data), 1)
    sl2go = StratifiedLeave2GroupsOut()
    clf = LDA(solver=SOLVER)
    save = classification(clf, sl2go, data, labels, groups, N_PERM, n_jobs=n_jobs)

    savemat(save_file_path, save)


if __name__ == "__main__":
    Parallel(n_jobs=-1)(
        delayed(classif_psd)(st, el, fr, n_jobs=-1)
        for st, el, fr in product(STATE_LIST, CHANNEL_NAMES, list(FREQ_DICT.keys()))
    )
