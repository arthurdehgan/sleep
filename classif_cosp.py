"""Load crosspectrum matrix, perform classif, perm test, saves results.

Outputs one file per freq x state

Author: Arthur Dehgan"""
from time import time
from itertools import product
from scipy.io import savemat, loadmat
from path import Path as path
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

SAVE_PATH = path("")
RESULT_FILE_PATH = SAVE_PATH / "results"
if not RESULT_FILE_PATH.check():
    RESULT_FILE_PATH.mkdir()
FREQ_DICT = {}
STATE_LIST = []
WINDOW = 500
OVERLAP = 0
LABEL_PATH = path("")
PREFIX = "classif_"
NAME = "cosp"
PREFIX_LIST = PREFIX.split("_")
PERM = "perm" in PREFIX_LIST
N_PERM = 99 if PERM else None

print(NAME, PREFIX)


def main(state, freq):
    """Where the magic happens"""
    print(state, freq)
    labels = loadmat(LABEL_PATH / state + "_labels.mat")["y"].ravel()
    labels, groups = create_groups(labels)

    file_path = (
        RESULT_FILE_PATH / PREFIX
        + NAME
        + "_{}_{}_{}_{:.2f}.mat".format(state, freq, WINDOW, OVERLAP)
    )

    if not file_path.isfile():
        file_name = NAME + "_{}_{}_{}_{:.2f}.mat".format(state, freq, WINDOW, OVERLAP)
        data_file_path = SAVE_PATH / file_name

        if data_file_path.isfile():
            data_og = loadmat(data_file_path)
            data = prepare_data(data_og)

            crossval = StratifiedLeave2GroupsOut()
            lda = LDA()
            clf = TSclassifier(clf=lda)
            save = classification(
                clf, crossval, data, labels, groups, N_PERM, n_jobs=-1
            )

            print(save["acc_score"])

            savemat(file_path, save)

            print(
                "accuracy for %s %s : %0.2f (+/- %0.2f)"
                % (state, freq, np.mean(save["acc_score"]), np.std(save["acc"]))
            )
            if PERM:
                print("pval = {}".format(save["acc_pvalue"]))

        else:
            print(data_file_path.name + " Not found")


if __name__ == "__main__":
    TIMELAPSE_START = time()
    for freq, state in product(FREQ_DICT, STATE_LIST):
        main(state, freq)
    print("total time lapsed : %s" % elapsed_time(TIMELAPSE_START, time()))
