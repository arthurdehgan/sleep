"""Load crosspectrum matrix, perform classif, perm test, saves results.

Outputs one file per freq x state

Author: Arthur Dehgan"""
from time import time
from itertools import product
from scipy.io import savemat, loadmat
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold
from pyriemann.classification import TSclassifier
from utils import (
    create_groups,
    StratifiedLeave2GroupsOut,
    elapsed_time,
    prepare_data,
    classification,
)
from params import (
    SAVE_PATH,
    FREQ_DICT,
    STATE_LIST,
    WINDOW,
    OVERLAP,
    LABEL_PATH,
    CHANNEL_NAMES,
)

# import pdb

PREFIX = "classif_reduced_"
NAME = "cosp"
PREF_LIST = PREFIX.split("_")
REDUCED = "reduced" in PREF_LIST
FULL_TRIAL = "ft" in PREF_LIST or "moy" in PREF_LIST
SUBSAMPLE = "subsamp" in PREF_LIST
PERM = "perm" in PREF_LIST
N_PERM = 990 if PERM else None

SAVE_PATH = SAVE_PATH / NAME
print(NAME, PREFIX)


def backward_selection(
    clf, data, labels, cv=3, groups=None, prev_ind=None, prev_score=0, index_list=[]
):
    # Exit condition: we have tried everything
    if prev_ind == -1:
        return index_list, prev_score

    if prev_ind is None:
        ind = data.shape[1] - 1
    else:
        ind = prev_ind

    if isinstance(cv, int):
        index = np.random.permutation(list(range(len(data))))
        labels = labels[index]
        data = data[index]
        croval = StratifiedKFold(n_splits=cv)
    else:
        croval = cv

    # Do classification
    save = classification(clf, cv=cv, X=data, y=labels, groups=groups, n_jobs=-1)
    score = save["acc_score"]

    # removing ind from features
    reduced_data = []
    for submat in data:
        temp_a = np.delete(submat, ind, 0)
        temp_b = np.delete(temp_a, ind, 1)
        reduced_data.append(temp_b)
    reduced_data = np.asarray(reduced_data)
    # reduced_data = np.concatenate((data[:, :ind], data[:, ind+1:]), axis=1)

    # If better score we continue exploring this reduced data
    print(data.shape, ind, score, prev_score)
    if score >= prev_score:
        if prev_ind is None and ind == data.shape[1] - 1:
            ind = prev_ind
        index_list.append(ind)
        return backward_selection(
            clf,
            reduced_data,
            labels,
            croval,
            groups,
            prev_score=score,
            index_list=index_list,
        )

    # Else we use the same data but we delete the next index
    return backward_selection(
        clf,
        data,
        labels,
        croval,
        groups,
        prev_ind=ind - 1,
        prev_score=prev_score,
        index_list=index_list,
    )


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
            final_save = {}

            random_seed = 0
            data = loadmat(data_file_path)
            if FULL_TRIAL:
                data = data["data"]
            elif SUBSAMPLE:
                data = prepare_data(data, n_trials=n_trials, random_state=random_seed)
            else:
                data = prepare_data(data)

            sl2go = StratifiedLeave2GroupsOut()
            lda = LDA()
            clf = TSclassifier(clf=lda)
            best_combin, best_score = backward_selection(
                clf, data, labels, sl2go, groups
            )

            final_save = {
                "best_combin_index": best_combin,
                "best_combin": CHANNEL_NAMES[best_combin],
                "score": best_score,
            }
            savemat(file_path, final_save)

            print(f"Best combin: {CHANNEL_NAMES[best_combin]}, score: {best_score}")

        else:
            print(data_file_path.NAME + " Not found")


if __name__ == "__main__":
    TIMELAPSE_START = time()
    for freq, state in product(FREQ_DICT, STATE_LIST):
        main(state, freq)
    print("total time lapsed : %s" % elapsed_time(TIMELAPSE_START, time()))
