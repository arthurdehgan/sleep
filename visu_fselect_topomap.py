"""Generate topomaps"""
from mne.viz import plot_topomap
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from params import SAVE_PATH, STATE_LIST, CHANNEL_NAMES

plt.switch_backend("agg")

DATA_PATH = SAVE_PATH / "psd"
TTEST_RESULTS_PATH = DATA_PATH / "results"
RESULTS_PATH = DATA_PATH / "results/"
POS_FILE = SAVE_PATH / "../Coord_EEG_1020.mat"
SENSORS_POS = loadmat(POS_FILE)["Cor"]
# FREQS = ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta', 'Gamma1', 'Gamma2']
FREQS = ["Delta", "Theta", "Alpha", "Sigma", "Beta", "Gamma1"]
WINDOW = 1000
OVERLAP = 0
PVAL = .01

for stage in STATE_LIST:

    for freq in FREQS:
        efs_scores, pvalues = [], []
        og_scores = []
        for elec in CHANNEL_NAMES:
            file_name = "EFS_NoGamma_{}_{}_1000_0.00.mat".format(stage, elec)
            try:
                score = loadmat(RESULTS_PATH / file_name)["score"].ravel()
            except TypeError:
                print(file_name)
            except KeyError:
                print("wrong key")
            efs_scores.append(score[0] * 100)

            file_name = "perm_PSD_{}_{}_{}_{}_{:.2f}.mat".format(
                stage, freq, elec, WINDOW, OVERLAP
            )
            try:
                score = loadmat(RESULTS_PATH / file_name)["score"].ravel()
            except TypeError:
                print(file_name)
            og_scores.append(score[0] * 100)

            pvalue = loadmat(RESULTS_PATH / file_name)["pvalue"].ravel()
            pvalues.append(pvalue[0])

        EFS_DA = np.asarray(efs_scores)
        OG_DA = np.asarray(og_scores)
        da_pvalues = np.asarray(pvalues)

        efs_mask = np.full((len(CHANNEL_NAMES)), False, dtype=bool)
        #         if freq == 'Delta':
        da_mask = np.full((len(CHANNEL_NAMES)), False, dtype=bool)
        da_mask[da_pvalues <= PVAL] = True
        mask_params = dict(
            marker="*", markerfacecolor="white", markersize=9, markeredgecolor="white"
        )

        #         data = [{'name': 'Decoding accuracies', 'cmap': 'viridis',
        #                  'mask': da_mask, 'cbarlim': [50, 65], 'data': OG_DA}]
        data = [
            {
                "name": "EFS decoding accuracies",
                "cmap": "viridis",
                "mask": da_mask,
                "cbarlim": [50, 65],
                "data": EFS_DA,
            }
        ]

        for i, subset in enumerate(data):
            plt.subplot(1, len(data), i + 1)
            ch_show = True
            ax, _ = plot_topomap(
                subset["data"],
                SENSORS_POS,
                res=128,
                cmap=subset["cmap"],
                show=False,
                vmin=subset["cbarlim"][0],
                vmax=subset["cbarlim"][1],
                names=CHANNEL_NAMES,
                show_names=ch_show,
                mask=subset["mask"],
                mask_params=mask_params,
                contours=0,
            )
            plt.colorbar(ax, shrink=.45)

        #         file_name = 'topomap_{}_mean_scores_EFS_{}'.format(stage, freq)
        file_name = "topomap_mean_scores_EFS_{}".format(stage)
        plt.savefig(SAVE_PATH / "../figures" / file_name, dpi=200)
        plt.close()
        del ax, data
