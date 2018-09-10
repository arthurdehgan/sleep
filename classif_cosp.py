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
from utils import (
    create_groups,
    StratifiedLeave2GroupsOut,
    elapsed_time,
    prepare_data,
    classification,
)
from params import SAVE_PATH, FREQ_DICT, STATE_LIST, WINDOW, OVERLAP, LABEL_PATH

# PREFIX = 'perm_'
# PREFIX = 'classif_'
PREFIX = "reduced_classif_"
# PREFIX = "bootstrapped_classif_"
# NAME = "subsamp_cosp"
NAME = "cosp"
# NAME = 'ft_cosp'
# NAME = 'moy_cosp'
# NAME = 'im_cosp'
# NAME = 'wpli'
# NAME = 'coh'
# NAME = 'imcoh'
# NAME = 'ft_wpli'
# NAME = 'ft_coh'
# NAME = 'ft_imcoh'
PREFIX_LIST = PREFIX.split("_")
BOOTSTRAP = "bootstrapped" in PREFIX_LIST
REDUCED = "reduced" in PREFIX_LIST
FULL_TRIAL = "ft" in PREFIX_LIST or "moy" in PREFIX_LIST
SUBSAMPLE = "subsamp" in NAME.split("_")
PERM = "perm" in PREFIX_LIST
N_PERM = 990 if PERM else None
if BOOTSTRAP:
    N_BOOTSTRAPS = 100
elif REDUCED:
    N_BOOTSTRAPS = 19
else:
    N_BOOTSTRAPS = 1

SAVE_PATH = SAVE_PATH / NAME
print(NAME, PREFIX)


def main(state, freq):
    """Where the magic happens"""
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
    else:
        labels = loadmat(LABEL_PATH / state + "_labels.mat")["y"].ravel()
        labels, groups = create_groups(labels)

    file_path = (
        SAVE_PATH / "results" / PREFIX
        + NAME
        + "_{}_{}_{}_{:.2f}.mat".format(state, freq, WINDOW, OVERLAP)
    )

    if not file_path.isfile():
        file_name = NAME + "_{}_{}_{}_{:.2f}.mat".format(state, freq, WINDOW, OVERLAP)
        data_file_path = SAVE_PATH / file_name

        if data_file_path.isfile():
            data_og = loadmat(data_file_path)
            for i in range(N_BOOTSTRAPS):
                if FULL_TRIAL:
                    data = data_og["data"]
                elif SUBSAMPLE:
                    data = prepare_data(data_og, n_trials=n_trials, random_state=i)
                else:
                    data = prepare_data(data_og)

                if REDUCED:
                    reduced_data = []
                    for submat in data:
                        temp_a = np.delete(submat, i, 0)
                        temp_b = np.delete(temp_a, i, 1)
                        reduced_data.append(temp_b)
                    data = np.asarray(reduced_data)

                sl2go = StratifiedLeave2GroupsOut()
                lda = LDA()
                clf = TSclassifier(clf=lda)
                save = classification(
                    clf, sl2go, data, labels, groups, N_PERM, n_jobs=-1
                )

                print(save["acc_score"])
                if BOOTSTRAP or REDUCED:
                    if i == 0:
                        final_save = save
                    else:
                        for key, value in save.items():
                            final_save[key] += value

            final_save["n_rep"] = N_BOOTSTRAPS
            if BOOTSTRAP:
                final_save["auc_score"] = np.mean(final_save["auc_score"])
                final_save["acc_score"] = np.mean(final_save["acc_score"])
            savemat(file_path, final_save)

            print(
                "accuracy for %s %s : %0.2f (+/- %0.2f)"
                % (state, freq, save["acc_score"], np.std(save["acc"]))
            )

        else:
            print(data_file_path.name + " Not found")


if __name__ == "__main__":
    TIMELAPSE_START = time()
    for freq, state in product(FREQ_DICT, STATE_LIST):
        main(state, freq)
    print("total time lapsed : %s" % elapsed_time(TIMELAPSE_START, time()))
