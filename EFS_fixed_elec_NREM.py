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
from params import SAVE_PATH, CHANNEL_NAMES, WINDOW, OVERLAP, STATE_LIST, FREQ_DICT

N_PERM = 1000
PERM = False
SAVE_PATH = SAVE_PATH / "psd"
if "Gamma1" in FREQ_DICT:
    del FREQ_DICT["Gamma1"]
FREQS = np.array(list(FREQ_DICT.keys()))
print(FREQS)
STATE = "NREM"


def main(elec):
    """feature selection and permutations.

    For each separation of subjects with leave 2 subjects out, we train on the
    big set (feature selection) and test on the two remaining subjects.
    for each permutation, we just permute the labels at the trial level (we
    could use permutations at the subject level, but we wouldn't get as many
    permutations)
    """
    final_data = None

    print(STATE, elec)
    results_file_path = (
        SAVE_PATH
        / "results"
        / "EFS_NoGamma_{}_{}_{}_{:.2f}.mat".format(STATE, elec, WINDOW, OVERLAP)
    )
    if not results_file_path.isfile():
        for freq in FREQS:
            data_file_path = SAVE_PATH / "PSD_{}_{}_{}_{}_{:.2f}.mat".format(
                STATE, freq, elec, WINDOW, OVERLAP
            )

            data = loadmat(data_file_path)["data"].ravel()
            if final_data is None:
                final_data = data
            else:
                for i, submat in enumerate(final_data):
                    final_data[i] = np.concatenate((submat, data[i]), axis=0)

        final_data = np.array(list(map(np.transpose, final_data)))

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

            x_train, x_test = final_data[train_subjects], final_data[test_subjects]
            y_train, y_test = lil_labels[train_subjects], lil_labels[test_subjects]

            y_train = [[label] * len(x_train[i]) for i, label in enumerate(y_train)]
            y_train, groups = create_groups(y_train)
            x_train = np.concatenate(x_train[:], axis=0)

            nested_sl2go = StratifiedLeave2GroupsOut()
            clf = LDA()
            f_select = EFS(
                estimator=clf,
                max_features=x_train.shape[-1],
                cv=nested_sl2go,
                n_jobs=-1,
            )

            f_select = f_select.fit(x_train, y_train, groups)

            best_idx = f_select.best_idx_
            best_freqs.append(list(FREQS[list(best_idx)]))
            best_scores.append(f_select.best_score_)

            test_clf = LDA()
            test_clf.fit(x_train[:, best_idx], y_train)
            y_test = [[label] * len(x_test[i]) for i, label in enumerate(y_test)]
            y_test, groups = create_groups(y_test)
            x_test = np.concatenate(x_test[:], axis=0)
            test_score = test_clf.score(x_test[:, best_idx], y_test)
            test_scores.append(test_score)

            if PERM:
                pscores_cv = []
                for _ in range(N_PERM):
                    y_train = np.random.permutation(y_train)
                    y_test = np.random.permutation(y_test)

                    clf = LDA()
                    clf.fit(x_train[:, best_idx], y_train)
                    pscore = clf.score(x_test[:, best_idx], y_test)
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

    for elec in CHANNEL_NAMES:
        main(elec)

    print("total time lapsed : {}".format(elapsed_time(T0, time())))
