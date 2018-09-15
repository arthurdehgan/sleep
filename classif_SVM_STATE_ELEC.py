"""Uses a classifier to decode PSD values.

Computes pvalues and saves them in a mat format with the decoding accuracies.

Author: Arthur Dehgan
"""
import numpy as np
import sys
from scipy.io import savemat, loadmat
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV as RS
from utils import StratifiedLeave2GroupsOut, create_groups, prepare_data
from params import SAVE_PATH, LABEL_PATH, WINDOW, OVERLAP, FREQ_DICT

SAVE_PATH = SAVE_PATH / "psd"
PREFIX = "classif_svm_"
N_PERM = None
STATE, ELEC = sys.argv[1], sys.argv[2]

LABELS = loadmat(LABEL_PATH / STATE + "_labels.mat")["y"].ravel()
LABELS, GROUPS = create_groups(LABELS)

for freq in FREQ_DICT:
    print(STATE, ELEC, freq)

    data_file_name = "PSD_{}_{}_{}_{}_{:.2f}.mat".format(
        STATE, freq, ELEC, WINDOW, OVERLAP
    )
    save_file_name = PREFIX + data_file_name
    data_file_path = SAVE_PATH / data_file_name
    save_file_path = SAVE_PATH / "results" / save_file_name

    if not save_file_path.isfile():
        data = loadmat(data_file_path)
        data = prepare_data(data)
        data = np.array(data).reshape(len(data), 1)

        cross_val = StratifiedLeave2GroupsOut()
        save = {"score": [], "cv_results": [], "best_score": [], "best_params": []}
        for train_index, test_index in cross_val.split(data, LABELS, GROUPS):
            train_set, validation_set = data[train_index], data[test_index]
            train_labs, validation_labs = LABELS[train_index], LABELS[test_index]
            train_groups, validation_groups = GROUPS[train_index], GROUPS[test_index]

            nested_cv = StratifiedLeave2GroupsOut()
            clf = SVC(kernel="rbf")
            parameters = {"C": np.logspace(-3, 2, 6), "gamma": np.logspace(-3, 2, 6)}
            random_search = RS(clf, parameters, n_iter=10, n_jobs=-1, cv=nested_cv)
            random_search = random_search.fit(
                X=train_set, y=train_labs, groups=train_groups
            )
            save["score"].append(random_search.score(validation_set, validation_labs))
            save["cv_results"].append(random_search.cv_results_)
            save["best_score"].append(random_search.best_score_)
            save["best_params"].append(random_search.best_params_)

        savemat(save_file_path, save)

        print("{} : {:.2f}".format(freq, save["score"]))
