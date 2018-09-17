"""Load covariance matrix, perform classif, perm test, saves results.

Outputs one file per freq x cond

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

prefix = "classif_"
name = "cov"

SAVE_PATH = ""
COND_LIST = ""
LABEL_PATH = ""
pref_list = prefix.split("_")
PERM = "perm" in pref_list
N_PERM = 999 if PERM else None

SAVE_PATH = SAVE_PATH / name


def main(cond):
    """Where the magic happens"""
    print(cond)
    labels = loadmat(LABEL_PATH / cond + "_labels.mat")["y"].ravel()
    labels, groups = create_groups(labels)

    file_name = prefix + name + "_{}.mat".format(cond)

    save_file_path = SAVE_PATH / "results" / file_name

    if not save_file_path.isfile():
        data_file_path = SAVE_PATH / name + "_{}.mat".format(cond)

        if data_file_path.isfile():
            data = loadmat(data_file_path)
            data = prepare_data(data)

            sl2go = StratifiedLeave2GroupsOut()
            lda = LDA()
            clf = TSclassifier(clf=lda)
            save = classification(clf, sl2go, data, labels, groups, N_PERM, n_jobs=-1)

            savemat(save_file_path, save)

            print(
                "accuracy for %s : %0.2f (+/- %0.2f)"
                % (cond, save["acc_score"], np.std(save["acc"]))
            )

        else:
            print(data_file_path.name + " Not found")


if __name__ == "__main__":
    TIMELAPSE_START = time()
    for cond in COND_LIST:
        main(cond)
    print("total time lapsed : %s" % elapsed_time(TIMELAPSE_START, time()))
