"""Loads results from EFS and adds permutations to the savefile.

Author: Arthur Dehgan
"""
from itertools import product
import numpy as np
from scipy.io import savemat, loadmat
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from utils import StratifiedLeave2GroupsOut, create_groups, compute_pval
from params import SAVE_PATH, CHANNEL_NAMES, WINDOW, OVERLAP, STATE_LIST, FREQ_DICT


N_PERM = 1000
SAVE_PATH = SAVE_PATH / "psd"
RESULT_PATH = SAVE_PATH / "results"
if "Gamma1" in FREQ_DICT:
    del FREQ_DICT["Gamma1"]
FREQS = list(FREQ_DICT.keys())


def load_data(state, elec):
    """Loads data for state, elec parameters."""
    final_data = None
    for freq in FREQS:
        data_file_path = (
            SAVE_PATH / f"PSD_{state}_{freq}_{elec}_{WINDOW}_{OVERLAP:.2f}.mat"
        )

        data = loadmat(data_file_path)["data"].ravel()
        if final_data is None:
            final_data = data
        else:
            for i, submat in enumerate(final_data):
                final_data[i] = np.concatenate((submat, data[i]), axis=0)
    return final_data


def main(state, elec):
    """Permutations.

    For each separation of subjects with leave 2 subjects out, we train on the
    big set and test on the two remaining subjects.
    for each permutation, we just permute the labels at the trial level (we
    could use permutations at the subject level, but we wouldn't get as many
    permutations)
    """
    file_name = f"EFS_NoGamma_{state}_{elec}_{WINDOW}_{OVERLAP:.2f}.mat"
    print(file_name)
    file_path = RESULT_PATH / file_name
    data = loadmat(file_path)

    lil_labels = [0] * 18 + [1] * 18
    lil_labels = np.asarray(lil_labels)
    lil_groups = list(range(36))
    sl2go = StratifiedLeave2GroupsOut()

    best_freqs = list(data["freqs"].ravel())
    scores = list(data["test_scores"].ravel())

    data = load_data(state, elec)
    pscores = []
    pvalues = []
    i = 0
    for train_subjects, test_subjects in sl2go.split(data, lil_labels, lil_groups):
        x_feature, x_classif = data[train_subjects], data[test_subjects]
        y_feature = lil_labels[train_subjects]
        y_classif = lil_labels[test_subjects]

        y_feature = [
            np.array([label] * x_feature[i].shape[1])
            for i, label in enumerate(y_feature)
        ]
        y_feature, _ = create_groups(y_feature)
        y_classif = [
            np.array([label] * x_classif[i].shape[1])
            for i, label in enumerate(y_classif)
        ]
        y_classif, _ = create_groups(y_classif)

        print(best_freqs[i])
        best_idx = [FREQS.index(value.strip().capitalize()) for value in best_freqs[i]]
        x_classif = np.concatenate(x_classif[:], axis=1).T
        x_feature = np.concatenate(x_feature[:], axis=1).T

        for _ in range(N_PERM):
            y_feature = np.random.permutation(y_feature)
            y_classif = np.random.permutation(y_classif)

            clf = LDA()
            clf.fit(x_feature[:, best_idx], y_feature)
            pscore = clf.score(x_classif[:, best_idx], y_classif)
            pscores.append(pscore)

        score = scores[i]
        pvalue = compute_pval(score, pscores)
        pvalues.append(pvalue)
        i += 1

    data["pvalue"] = pvalues
    data["pscores"] = pscores

    savemat(file_path, data)


if __name__ == "__main__":
    for state, elec in product(STATE_LIST, CHANNEL_NAMES):
        main(state, elec)
