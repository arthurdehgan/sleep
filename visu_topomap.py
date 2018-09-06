"""Generate topomaps"""
from mne.viz import plot_topomap
from scipy.io import loadmat
from scipy.stats import zscore
from params import SAVE_PATH, STATE_LIST, CHANNEL_NAMES
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend("agg")

DATA_PATH = SAVE_PATH / "psd"
TTEST_RESULTS_PATH = DATA_PATH / "results"
solver = "svd"
RESULTS_PATH = DATA_PATH / "results/"
POS_FILE = SAVE_PATH / "../Coord_EEG_1020.mat"
SENSORS_POS = loadmat(POS_FILE)["Cor"]
# FREQS = ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta', 'Gamma1', 'Gamma2']
FREQS = ["Delta", "Theta", "Alpha", "Sigma", "Beta"]
# prefix = 'bootstrapped_perm_subsamp_'
prefix = "perm_"
WINDOW = 1000
OVERLAP = 0
p = .01

for stage in STATE_LIST:

    k, j = 1, 1
    fig = plt.figure(figsize=(14, 12))
    for freq in FREQS:

        scores, pvalues = [], []
        dreamer, ndreamer = [], []
        for elec in CHANNEL_NAMES:
            file_name = prefix + "PSD_{}_{}_{}_{}_{:.2f}.mat".format(
                stage, freq, elec, WINDOW, OVERLAP
            )
            try:
                score = loadmat(RESULTS_PATH / file_name)
                score = score["score"].ravel().mean()
                # score = score['acc_score'].ravel().mean()
                pvalue = loadmat(RESULTS_PATH / file_name)["pvalue"].ravel()
                # pvalue = loadmat(RESULTS_PATH / file_name)['acc_pvalue'].ravel()
            except TypeError:
                score = [.5]
                pvalue = [1]
                print(file_name)
            scores.append(score * 100)
            pvalues.append(pvalue[0])

            file_name = "PSD_{}_{}_{}_{}_{:.2f}.mat".format(
                stage, freq, elec, WINDOW, OVERLAP
            )
            try:
                PSD = loadmat(DATA_PATH / file_name)["data"].ravel()
            except TypeError:
                print(file_name)
            ndreamer.append(np.mean([e.ravel().mean() for e in PSD[18:]]))
            dreamer.append(np.mean([e.ravel().mean() for e in PSD[:18]]))

        ttest = loadmat(TTEST_RESULTS_PATH / "ttest_perm_{}_{}.mat".format(stage, freq))
        tt_pvalues = ttest["p_values"].ravel()
        t_values = zscore(ttest["t_values"].ravel())
        dreamer = np.asarray(dreamer)
        ndreamer = np.asarray(ndreamer)
        DA = np.asarray(scores)
        da_pvalues = np.asarray(pvalues)
        RPC = zscore((dreamer - ndreamer) / ndreamer)
        dreamer = zscore(dreamer)
        ndreamer = zscore(ndreamer)

        da_mask = np.full((len(CHANNEL_NAMES)), False, dtype=bool)
        tt_mask = np.full((len(CHANNEL_NAMES)), False, dtype=bool)
        tt_mask[tt_pvalues <= p] = True
        da_mask[da_pvalues <= p] = True
        mask_params = dict(
            marker="*", markerfacecolor="white", markersize=9, markeredgecolor="white"
        )

        data = [
            {
                "name": "PSD High Recallers",
                "cmap": "jet",
                "mask": None,
                "cbarlim": [min(dreamer), max(dreamer)],
                "data": dreamer / max(dreamer),
            },
            {
                "name": "PSD Low Recallers",
                "cmap": "jet",
                "mask": None,
                "cbarlim": [min(ndreamer), max(ndreamer)],
                "data": ndreamer / max(ndreamer),
            },
            {
                "name": "Relative Power Changes",
                "cmap": "inferno",
                "mask": None,
                "cbarlim": [min(RPC), max(RPC)],
                "data": RPC / max(RPC),
            },
            {
                "name": "ttest permutations p<{}".format(p),
                "data": t_values / max(t_values),
                "cmap": "viridis",
                "mask": tt_mask,
                "cbarlim": [min(t_values), max(t_values)],
            },
            {
                "name": "Decoding Accuracies p<{}".format(p),
                "cmap": "viridis",
                "mask": da_mask,
                "cbarlim": [50, 65],
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
                # fig.colorbar(ax, orientation='horizontal')
            if i == 0:
                plt.ylabel(freq)

        j += 1
        k += 5

    plt.subplots_adjust(
        left=None, bottom=0.05, right=None, top=None, wspace=None, hspace=None
    )
    plt.tight_layout()
    file_name = "noco_topomap_{}_{}_p{}".format(solver, stage, str(p)[2:])
    plt.savefig(SAVE_PATH / "../figures" / file_name, dpi=200)
