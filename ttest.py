"""Ttest for PSD values.
by Arthur Dehgan"""

from scipy.io import loadmat, savemat
import numpy as np
from joblib import Parallel, delayed
from ttest_perm_indep import ttest_perm_unpaired
from params import STATE_LIST, FREQ_DICT, SAVE_PATH, WINDOW, OVERLAP, CHANNEL_NAMES

SAVE_PATH = SAVE_PATH / "psd"
RESULT_PATH = SAVE_PATH / "results"
n_perm = 9999


def main(stade, freq):
    HRs, LRs = [], []
    for elec in CHANNEL_NAMES:

        file_path = SAVE_PATH / "PSD_{}_{}_{}_{}_{:.2f}.mat".format(
            stade, freq, elec, WINDOW, OVERLAP
        )
        try:
            X = loadmat(file_path)["data"].ravel()
        except KeyError:
            print(file_path, "corrupted")
        except IOError:
            print(file_path, "Not Found")
        X = np.delete(X, 9, 0)  # delete subj 10 cuz of artefact on FC2
        HR = X[:17]
        LR = X[17:]
        HR = np.concatenate([psd.flatten() for psd in HR])
        LR = np.concatenate([psd.flatten() for psd in LR])
        # for i in range(len(HR)):
        #     HR[i] = HR[i].mean()
        #     LR[i] = LR[i].mean()

        HRs.append(HR)
        LRs.append(LR)

    HRs = np.asarray(HRs, dtype=float).T
    LRs = np.asarray(LRs, dtype=float).T
    tval, pvalues = ttest_perm_unpaired(
        cond1=HRs, cond2=LRs, n_perm=n_perm, equal_var=False, two_tailed=True, n_jobs=-2
    )

    data = {"p_values": np.asarray(pvalues), "t_values": tval}
    file_path = RESULT_PATH / "ttest_perm_{}_{}.mat".format(stade, freq)
    savemat(file_path, data)


if __name__ == "__main__":
    Parallel(n_jobs=1)(
        delayed(main)(stade, freq) for stade in STATE_LIST for freq in FREQ_DICT
    )
