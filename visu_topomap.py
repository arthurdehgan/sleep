"""Generate topomaps"""
from mne.viz import plot_topomap
from scipy.io import loadmat
from scipy.stats import zscore, binom
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import compute_pval

# from matplotlib import ticker
from params import SAVE_PATH, STATE_LIST, CHANNEL_NAMES

plt.switch_backend("agg")

DATA_PATH = SAVE_PATH / "psd"
TTEST_RESULTS_PATH = DATA_PATH / "results"
RESULTS_PATH = DATA_PATH / "results/"
POS_FILE = SAVE_PATH / "../Coord_EEG_1020.mat"
SENSORS_POS = loadmat(POS_FILE)["Cor"]
# FREQS = ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta', 'Gamma1', 'Gamma2']
FREQS = ["Delta", "Theta", "Alpha", "Sigma", "Beta"]
PREFIX = "bootstrapped_perm_subsamp_"
# PREFIX = "perm_"
SUBSAMP = "subsamp" in PREFIX.split("_")
WINDOW = 1000
OVERLAP = 0
PVAL = .01
BINOM = False
CORRECTION = True
INFO_DATA = pd.read_csv(SAVE_PATH / "info_data.csv")[STATE_LIST]
TRIALS = list(INFO_DATA.iloc[36])

for stage in STATE_LIST:

    k, j = 1, 1
    fig = plt.figure(figsize=(8, 10))
    for freq in FREQS:

        scores, pscores_all_elec = [], []
        HR, LR = [], []
        for elec in CHANNEL_NAMES:
            file_name = PREFIX + "PSD_{}_{}_{}_{}_{:.2f}.mat".format(
                stage, freq, elec, WINDOW, OVERLAP
            )
            try:
                results = loadmat(RESULTS_PATH / file_name)
                # if SUBSAMP:
                score_key = "acc"
                pscores_key = "acc_pscores"
                # else:
                #     score_key = "score"
                #     pscores_key = "pscore"
                score = float(results[score_key].ravel().mean())
                pscores = list(results[pscores_key].squeeze())
            except TypeError as error:
                score = [.5]
                pvalue = [1]
                print(error, file_name)
            except:
                print("Error: ", file_name)

            scores.append(score)
            pscores_all_elec += pscores

            file_name = "PSD_{}_{}_{}_{}_{:.2f}.mat".format(
                stage, freq, elec, WINDOW, OVERLAP
            )
            try:
                PSD = loadmat(DATA_PATH / file_name)["data"].ravel()
                # subject 10 has artefact on FC2, so we just remove it
                PSD = np.delete(PSD, 9, 0)
            except TypeError:
                print(file_name)
            HR.append(np.mean([e.ravel().mean() for e in PSD[:17]]))
            LR.append(np.mean([e.ravel().mean() for e in PSD[17:]]))

        pvalues = []
        for score in scores:
            pvalues.append(compute_pval(score, pscores_all_elec))

        ttest = loadmat(TTEST_RESULTS_PATH / "ttest_perm_{}_{}.mat".format(stage, freq))
        tt_pvalues = ttest["p_values"].ravel()
        t_values = zscore(ttest["t_values"].ravel())
        HR = np.asarray(HR)
        LR = np.asarray(LR)
        DA = 100 * np.asarray(scores)
        da_pvalues = np.asarray(pvalues)
        # RPC = zscore((HR - LR) / LR)
        # HR = HR / max(abs(HR))
        # LR = LR / max(abs(LR))
        # HR = zscore(HR)
        # LR = zscore(LR)

        da_mask = np.full((len(CHANNEL_NAMES)), False, dtype=bool)
        tt_mask = np.full((len(CHANNEL_NAMES)), False, dtype=bool)
        tt_mask[tt_pvalues <= PVAL] = True
        if BINOM:
            thresholds = [
                100 * binom.isf(PVAL, n_trials, .5) / n_trials for n_trials in TRIALS
            ]
            da_mask[DA > thresholds[j]] = True
        else:
            da_mask[da_pvalues <= PVAL] = True

        mask_params = dict(
            marker="*", markerfacecolor="white", markersize=9, markeredgecolor="white"
        )

        data = [
            {
                "name": "PSD HR",
                "cmap": "jet",
                "mask": None,
                "cbarlim": [min(HR), max(HR)],
                "data": HR,
            },
            {
                "name": "PSD LR",
                "cmap": "jet",
                "mask": None,
                "cbarlim": [min(LR), max(LR)],
                "data": LR,
            },
            # {
            #     "name": "Relative Power Changes",
            #     "cmap": "inferno",
            #     "mask": None,
            #     "cbarlim": [min(RPC), max(RPC)],
            #     "data": RPC / max(RPC),
            # },
            {
                "name": "corrected T-values",
                "data": t_values,
                "cmap": "viridis",
                "mask": tt_mask,
                "cbarlim": [min(t_values), max(t_values)],
            },
            {
                "name": "Decoding Accuracies (%)",
                "cmap": "viridis",
                "mask": da_mask,
                "cbarlim": [50, 60],
                "data": DA,
            },
        ]

        for i, subset in enumerate(data):
            plt.subplot(len(FREQS), len(data), i + k)
            ch_show = False if i > 1 else True
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
            if freq == FREQS[-1]:
                plt.xlabel(subset["name"])
            if freq == FREQS[-1]:
                pass
                # cb = fig.colorbar(ax, orientation="horizontal")
                # tick_locator = ticker.MaxNLocator(nbins=5)
                # cb.locator = tick_locator
                # cb.update_ticks()
            if i == 0:
                plt.ylabel(freq)

        j += 1
        k += len(data)

    plt.subplots_adjust(
        left=None, bottom=0.05, right=None, top=None, wspace=None, hspace=None
    )
    plt.tight_layout()
    file_name = "topomap_{}{}_p{}".format(PREFIX, stage, str(PVAL)[2:])
    print(file_name)
    plt.savefig(SAVE_PATH / "../figures" / file_name, dpi=400)
