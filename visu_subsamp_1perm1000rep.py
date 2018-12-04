"""Generate topomaps"""
from mne.viz import plot_topomap
from scipy.io import loadmat
from scipy.stats import zscore, binom
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from utils import compute_pval

# from matplotlib import ticker
from params import SAVE_PATH, STATE_LIST, CHANNEL_NAMES

plt.switch_backend("agg")

NAME = "psd"
PREFIX = "bootstrapped_subsamp_"
PERM_FILE_PREFIX = "perm_subsamp_"

DATA_PATH = SAVE_PATH / NAME
TTEST_RESULTS_PATH = DATA_PATH / "results"
RESULTS_PATH = DATA_PATH / "results/"
POS_FILE = SAVE_PATH / "../Coord_EEG_1020.mat"
INFO_DATA = pd.read_csv(SAVE_PATH / "info_data.csv")[STATE_LIST]
SENSORS_POS = loadmat(POS_FILE)["Cor"]

FREQS = ["Delta", "Theta", "Alpha", "Sigma", "Beta"]
SUBSAMP = "subsamp" in PREFIX.split("_")
BOOTSTRAPPED = "bootstrapped" in PREFIX.split("_")
WINDOW = 1000
OVERLAP = 0
PVAL = .001
MAXSTAT_ELEC = True
BINOM = False
INFO_DATA = pd.read_csv(SAVE_PATH / "info_data.csv")[STATE_LIST]
N_TRIALS = INFO_DATA.min().min()
TRIALS = list(INFO_DATA.iloc[36])


def gen_figure(stage):
    k, j = 1, 1
    fig = plt.figure(figsize=(8, 10))
    stds_tot = []
    for freq in FREQS:

        scores, pscores_all_elec = [], []
        stds = []
        HR, LR = [], []
        for elec in CHANNEL_NAMES:
            file_name = (
                PREFIX
                + NAME
                + "_{}_{}_{}_{}_{:.2f}.mat".format(stage, freq, elec, WINDOW, OVERLAP)
            )
            perm_file = (
                PERM_FILE_PREFIX
                + NAME
                + "_{}_{}_{}_{}_{:.2f}.mat".format(stage, freq, elec, WINDOW, OVERLAP)
            )
            results = loadmat(RESULTS_PATH / file_name)
            perm_f = loadmat(RESULTS_PATH / perm_file)
            score_key = "acc"
            pscores_key = "acc_pscores"
            acc_scores = results[score_key].ravel()
            score = float(acc_scores.mean())
            std_acc = acc_scores.std()
            pscores = list(perm_f[pscores_key].squeeze())
            n_rep = int(results["n_rep"])

            scores.append(score)
            stds.append(std_acc)
            pscores_all_elec.append(pscores)

            file_name = NAME + "_{}_{}_{}_{}_{:.2f}.mat".format(
                stage, freq, elec, WINDOW, OVERLAP
            )
            try:
                PSD = loadmat(DATA_PATH / file_name)["data"].ravel()
                moy_PSD = [0] * 36
                for i, submat in enumerate(PSD):
                    if BOOTSTRAPPED:
                        for random_state in range(n_rep):
                            index = np.random.RandomState(random_state).choice(
                                range(len(submat.ravel())), N_TRIALS, replace=False
                            )
                            prep_submat = submat.ravel()[index]
                            mean_for_the_sub = np.mean(prep_submat)
                            moy_PSD[i] += mean_for_the_sub / n_rep
                    else:
                        moy_PSD[i] = np.mean(submat)
                # subject 10 has artefact on FC2, so we just remove it
                moy_PSD = np.delete(moy_PSD, 9, 0)
            except TypeError:
                print(file_name)
            HR.append(np.mean(moy_PSD[:17]))
            LR.append(np.mean(moy_PSD[17:]))
            print(stage, freq, elec, ":", score, "+/-", std_acc)

        pscores_all_elec = np.asarray(pscores_all_elec)
        if MAXSTAT_ELEC:
            pscores_all_elec = np.max(pscores_all_elec, axis=0)

        pvalues = []
        for i, score in enumerate(scores):
            if MAXSTAT_ELEC:
                pscores = pscores_all_elec
            else:
                pscores = pscores_all_elec[i]
            pvalues.append(compute_pval(score, pscores))
        print(pvalues)

        ttest = loadmat(TTEST_RESULTS_PATH / "ttest_perm_{}_{}.mat".format(stage, freq))
        tt_pvalues = ttest["p_values"].ravel()
        t_values = zscore(ttest["t_values"].ravel())
        HR = np.asarray(HR)
        LR = np.asarray(LR)
        DA = 100 * np.asarray(scores)
        da_pvalues = np.asarray(pvalues)

        da_mask = np.full((len(CHANNEL_NAMES)), False, dtype=bool)
        tt_mask = np.full((len(CHANNEL_NAMES)), False, dtype=bool)
        tt_mask[tt_pvalues < PVAL] = True
        if BINOM:
            thresholds = [
                100 * binom.isf(PVAL, n_trials, .5) / n_trials for n_trials in TRIALS
            ]
            da_mask[DA > thresholds[j]] = True
        else:
            da_mask[da_pvalues < PVAL] = True

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
                "cbarlim": [50, 65],
                "data": DA,
            },
        ]

        for i, subset in enumerate(data):
            plt.subplot(len(FREQS), len(data), i + k)
            ch_show = False if i > 1 else True
            if subset["name"] == "Decoding Accuracies (%)":
                print(subset["data"])
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
        print(np.mean(stds))
        stds_tot.append(np.mean(stds))

    print(np.mean(stds_tot))
    plt.subplots_adjust(
        left=None, bottom=0.05, right=None, top=None, wspace=None, hspace=None
    )
    plt.tight_layout()
    file_name = "topomap_{}{}_{}_p{}".format(PREFIX, NAME, stage, str(PVAL)[2:])
    print(file_name)
    plt.savefig(SAVE_PATH / "../figures" / file_name, dpi=400)


if __name__ == "__main__":
    Parallel(n_jobs=1)(delayed(gen_figure)(stage) for stage in STATE_LIST)
