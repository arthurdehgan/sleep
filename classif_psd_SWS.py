"""Uses a classifier to decode PSD values.

Computes pvalues and saves them in a mat format with the decoding accuracies.

Author: Arthur Dehgan
"""
from time import time

# from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from itertools import product
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
    path,
    CHANNEL_NAMES,
    WINDOW,
    OVERLAP,
    STATE_LIST,
    FREQ_DICT,
)

# prefix = "perm"
# prefix = 'classif'
prefix = "bootstrapped_perm_subsamp_"
SOLVER = "svd"  # 'svd' 'lsqr'

pref_list = prefix.split("_")
BOOTSTRAP = "bootstrapped" in pref_list
SUBSAMPLE = "subsamp" in pref_list
PERM = "perm" in pref_list
N_PERM = 99 if PERM else None
N_BOOTSTRAPS = 100 if BOOTSTRAP else 1

SAVE_PATH = SAVE_PATH / "psd"
STATE = "SWS"


def main(elec):
    global SUBSAMPLE, SAVE_PATH
    if SUBSAMPLE:
        info_data = pd.read_csv(SAVE_PATH.parent / "info_data.csv")[STATE_LIST]
        N_TRIALS = info_data.min().min()
        N_SUBS = len(info_data) - 1
        groups = [i for i in range(N_SUBS) for _ in range(N_TRIALS)]
        N_TOTAL = N_TRIALS * N_SUBS
        labels = [0 if i < N_TOTAL / 2 else 1 for i in range(N_TOTAL)]
    else:
        labels = loadmat(LABEL_PATH / STATE + "_labels.mat")["y"].ravel()
        labels, groups = create_groups(labels)

    for freq in FREQ_DICT:
        print(STATE, elec, freq)

        data_file_name = "PSD_{}_{}_{}_{}_{:.2f}.mat".format(
            STATE, freq, elec, WINDOW, OVERLAP
        )

        save_file_name = prefix + data_file_name

        data_file_path = SAVE_PATH / data_file_name

        save_file_path = SAVE_PATH / "results" / save_file_name

        if not save_file_path.isfile():
            for i in range(N_BOOTSTRAPS):
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

                if BOOTSTRAP or REDUCED:
                    if i == 0:
                        final_save = save
                    else:
                        for key, value in save.items():
                            final_save[key] += value

            final_save["n_rep"] = N_BOOTSTRAPS
            final_save["auc_score"] = np.mean(final_save["auc_score"])
            final_save["acc_score"] = np.mean(final_save["acc_score"])
            savemat(save_file_path, final_save)

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
    for elec in CHANNEL_NAMES:
        main(elec)
    print("total time lapsed : %s" % (elapsed_time(TIMELAPSE_START, time())))
