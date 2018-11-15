"""Generate topomaps"""
from mne.viz import plot_topomap
from scipy.io import loadmat
from scipy.stats import zscore, binom
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import super_count
from params import SAVE_PATH, STATE_LIST, CHANNEL_NAMES

plt.switch_backend("agg")

DATA_PATH = SAVE_PATH / "psd"
RESULTS_PATH = DATA_PATH / "results/"
FREQS = ["Delta", "Theta", "Alpha", "Sigma", "Beta"]
REGS = ["Prefrontal", "Frontal", "Temporal", "Central", "Parietal", "Occipital"]
REGIONS = {
    "Occipital": ["O1", "O2"],
    "Prefrontal": ["Fp1", "Fp2"],
    "Fronto-Central": ["Cz", "C3", "C4", "FC1", "FC2", "F3", "F4", "Fz"],
    "Centro-Parietal": ["CP1", "CP2", "P4", "P3", "Pz"],
    "Temporal": ["T3", "T4"],
}

plt.figure(figsize=(8, 14))
for j, stage in enumerate(STATE_LIST):
    counts, all_count = {}, {}
    for elec in CHANNEL_NAMES:
        file_name = "EFS_NoGamma_{}_{}_1000_0.00.mat".format(stage, elec)
        freqs = loadmat(RESULTS_PATH / file_name)["freqs"].ravel()
        count = super_count(
            [freq.strip().capitalize() for sub in freqs for freq in sub]
        )
        counts[elec] = count
        for freq in FREQS:
            all_count[freq] = all_count.get(freq, 0) + count.get(freq, 0)

    plt.subplot2grid((7, 4), (0, j))
    plt.pie([all_count[freq] for freq in FREQS])
    if j == 0:
        plt.ylabel("All Stages")
    plt.xlabel(stage, verticalalignment="top")

    i = 1
    for region in REGS:
        elecs = REGIONS[region]
        sub_count = {}
        for freq in FREQS:
            sub_count[freq] = sum([counts[elec].get(freq, 0) for elec in elecs])
        plt.subplot2grid((7, 4), (i, j))
        plt.pie([sub_count[freq] for freq in FREQS])
        if j == 0:
            plt.ylabel(region)
        i += 1

file_name = "EFS_piechart"
print(file_name)
plt.legend(FREQS, loc="best")
plt.tight_layout()
plt.savefig(SAVE_PATH / "../figures" / file_name, dpi=200)
