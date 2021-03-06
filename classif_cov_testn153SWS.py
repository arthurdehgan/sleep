"""Load covariance matrix, perform classif, perm test, saves results.

Outputs one file per freq x state

Author: Arthur Dehgan"""
from time import time
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
from params import SAVE_PATH, STATE_LIST, LABEL_PATH

# import pdb

# name = 'moy_cov'
prefix = "classif_subsamp_"
name = "cov"

pref_list = prefix.split("_")
BOOTSTRAP = "bootstrapped" in pref_list
FULL_TRIAL = "ft" in pref_list or "moy" in pref_list
SUBSAMPLE = "subsamp" in pref_list
PERM = "perm" in pref_list
N_PERM = 999 if PERM else None
N_BOOTSTRAPS = 10 if BOOTSTRAP else 1

SAVE_PATH = SAVE_PATH / name

##### FOR A TEST #####
STATE_LIST = ["SWS"]
##### FOR A TEST #####


def main(state):
    """Where the magic happens"""
    print(state)
    if FULL_TRIAL:
        labels = np.concatenate((np.ones(18), np.zeros(18)))
        groups = range(36)
    elif SUBSAMPLE:
        info_data = pd.read_csv(SAVE_PATH.parent / "info_data.csv")[STATE_LIST]
        ##### FOR A TEST #####
        info_data = info_data["SWS"]
        ##### FOR A TEST #####
        N_TRIALS = info_data.min().min()
        N_SUBS = len(info_data) - 1
        groups = [i for _ in range(N_TRIALS) for i in range(N_SUBS)]
        N_TOTAL = N_TRIALS * N_SUBS
        labels = [0 if i < N_TOTAL / 2 else 1 for i in range(N_TOTAL)]
    else:
        labels = loadmat(LABEL_PATH / state + "_labels.mat")["y"].ravel()
        labels, groups = create_groups(labels)

    file_name = prefix + name + "n153_{}.mat".format(state)

    save_file_path = SAVE_PATH / "results" / file_name

    if not save_file_path.isfile():
        data_file_path = SAVE_PATH / name + "_{}.mat".format(state)

        if data_file_path.isfile():
            final_save = None

            for i in range(N_BOOTSTRAPS):
                data = loadmat(data_file_path)
                if FULL_TRIAL:
                    data = data["data"]
                elif SUBSAMPLE:
                    data = prepare_data(data, n_trials=N_TRIALS, random_state=i)
                else:
                    data = prepare_data(data)

                sl2go = StratifiedLeave2GroupsOut()
                lda = LDA()
                clf = TSclassifier(clf=lda)
                save = classification(
                    clf, sl2go, data, labels, groups, N_PERM, n_jobs=-1
                )
                save["acc_bootstrap"] = [save["acc_score"]]
                save["auc_bootstrap"] = [save["auc_score"]]
                if final_save is None:
                    final_save = save
                else:
                    for key, value in final_save.items():
                        final_save[key] = final_save[key] + save[key]

            savemat(save_file_path, final_save)

            print(
                "accuracy for %s : %0.2f (+/- %0.2f)"
                % (state, save["acc_score"], np.std(save["acc"]))
            )

        else:
            print(data_file_path.name + " Not found")


if __name__ == "__main__":
    TIMELAPSE_START = time()
    for state in STATE_LIST:
        main(state)
    print("total time lapsed : %s" % elapsed_time(TIMELAPSE_START, time()))
