"""Exaustive feature selection on frequencies for each electrode.

Computes pvalues and saves them in a mat format with the decoding accuracies.

Author: Arthur Dehgan
"""
from time import time
from itertools import product
import numpy as np
from scipy.io import savemat, loadmat

# from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from utils import StratifiedLeave2GroupsOut, elapsed_time, compute_pval, create_groups
from params import (
    SAVE_PATH,
    path,
    CHANNEL_NAMES,
    WINDOW,
    OVERLAP,
    STATE_LIST,
    FREQ_DICT,
)

N_PERM = 1000
SAVE_PATH = SAVE_PATH / "psd"
if "Gamma1" in FREQ_DICT:
    del FREQ_DICT["Gamma1"]
FREQS = np.array(list(FREQ_DICT.keys()))
print(FREQS)


def main(state, elec):
    """feature selection and permutations.

    For each separation of subjects with leave 2 subjects out, we train on the
    big set (feature selection) and test on the two remaining subjects.
    for each permutation, we just permute the labels at the trial level (we
    could use permutations at the subject level, but we wouldn't get as many
    permutations)
    """
    final_data = None

    print(state, elec)
    results_file_path = (
        SAVE_PATH
        / "results"
        / "EFS_NoGamma_{}_{}_{}_{:.2f}.mat".format(state, elec, WINDOW, OVERLAP)
    )
    if not path(results_file_path).isfile():
        # print('\nloading PSD for {} frequencies'.format(key))
        for key in FREQ_DICT:
            data_file_path = SAVE_PATH / "PSD_{}_{}_{}_{}_{:.2f}.mat".format(
                state, key, elec, WINDOW, OVERLAP
            )

            if path(data_file_path).isfile():
                data = loadmat(data_file_path)["data"].ravel()
                if final_data is None:
                    final_data = data
                else:
                    for i, submat in enumerate(final_data):
                        final_data[i] = np.concatenate((submat, data[i]), axis=0)
            else:
                print(path(data_file_path).name + " Not found")
                print(
                    'please run "computePSD.py" and\
                      "group_PSD_per_subjects.py" before\
                      running this script'
                )

        lil_labels = [0] * 18 + [1] * 18
        lil_labels = np.asarray(lil_labels)
        lil_groups = list(range(36))
        sl2go = StratifiedLeave2GroupsOut()

        best_freqs = []
        pvalues, pscores = [], []
        test_scores, best_scores = [], []
        for train_subjects, test_subjects in sl2go.split(
            final_data, lil_labels, lil_groups
        ):

            x_feature, x_classif = data[train_subjects], data[test_subjects]
            y_feature = lil_labels[train_subjects]
            y_classif = lil_labels[test_subjects]

            y_feature = [
                np.array([label] * x_feature[i].shape[1])
                for i, label in enumerate(y_feature)
            ]
            y_feature, groups = create_groups(y_feature)
            x_feature = np.concatenate(x_feature[:], axis=1).T

            nested_sl2go = StratifiedLeave2GroupsOut()
            clf = LDA()
            f_select = EFS(
                estimator=clf,
                max_features=x_feature.shape[-1],
                cv=nested_sl2go,
                verbose=0,
                n_jobs=-1,
            )

            f_select = f_select.fit(x_feature, y_feature, groups)

            best_idx = f_select.best_idx_
            best_freqs.append(FREQS[list(best_idx)])
            best_scores.append(f_select.best_score_)

            test_clf = LDA()
            test_clf.fit(x_feature[:, best_idx], y_feature)
            y_classif = [
                np.array([label] * x_classif[i].shape[1])
                for i, label in enumerate(y_classif)
            ]
            y_classif, groups = create_groups(y_classif)
            x_classif = np.concatenate(x_classif[:], axis=1).T
            test_score = test_clf.score(x_classif[:, best_idx], y_classif)
            test_scores.append(test_score)

            pscores_cv = []
            for _ in range(N_PERM):
                y_feature = np.random.permutation(y_feature)
                y_classif = np.random.permutation(y_classif)

                clf = LDA()
                clf.fit(x_feature[:, best_idx], y_feature)
                pscore = clf.score(x_classif[:, best_idx], y_classif)
                pscores_cv.append(pscore)

            pvalue = compute_pval(test_score, pscores_cv)
            pvalues.append(pvalue)
            pscores.append(pscores_cv)

        score = np.mean(test_scores)
        data = {
            "score": score,
            "train_scores": best_scores,
            "test_scores": test_scores,
            "freqs": best_freqs,
            "pvalue": pvalues,
            "pscores": pscores,
        }

        savemat(results_file_path, data)


if __name__ == "__main__":
    T0 = time()

    for state, elec in product(STATE_LIST, CHANNEL_NAMES):
        main(state, elec)

    print("total time lapsed : {}".format(elapsed_time(T0, time())))
