"""Boxplots of the data"""
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.stats import zscore
from itertools import product
from utils import rm_outliers
from params import STATE_LIST, FREQ_DICT, CHANNEL_NAMES, SAVE_PATH

SAVE_PATH /= "psd"
RANDOM = 666
N_SUBJ = 36
N_ELEC = len(CHANNEL_NAMES)

for st, fr in product(STATE_LIST, FREQ_DICT):
    PSD_all_elec = []
    first = True
    for ch in CHANNEL_NAMES:
        file_name = SAVE_PATH / f"psd_{st}_{fr}_{ch}_1000_0.00.mat"
        PSD = loadmat(file_name)["data"].ravel()
        for i, sub in enumerate(PSD):
            # clean_psd = rm_outliers(sub.ravel(), 2)
            clean_psd = sub.ravel()
            if first:
                PSD_all_elec.append(clean_psd / N_ELEC)
            else:
                PSD_all_elec[i] += clean_psd / N_ELEC

        first = False
        # sizes = []
        # for sub in PSD:
        #     sizes.append(len(sub.ravel()))
        # n_trials = min(sizes)
        # final = []
        # for i, submat in enumerate(PSD):
        #     index = np.random.RandomState(RANDOM).choice(
        #         range(len(submat.ravel())), n_trials, replace=False
        #     )
        #     chosen = submat.ravel()[index]
        #     final.append(chosen)

    # to set ylim we check the highest that is not outlier (check zscore)
    psdmax = 0
    for sub in PSD_all_elec:
        psd_max_c = rm_outliers(sub, 3).max()
        if psdmax < psd_max_c:
            psdmax = psd_max_c

    plt.figure(figsize=(15, 15))
    sns.boxplot(data=PSD_all_elec)
    plt.ylim(0, psdmax)
    plt.title(f"PSD per subject for {st}, {fr}")
    plt.tight_layout()
    plt.show()
