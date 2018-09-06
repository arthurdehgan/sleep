"""Generates the visu for a da matrix

Author: Arthur Dehgan"""
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.io import loadmat
from params import WINDOW, OVERLAP, FBIN_LIST, CHANNEL_NAMES, SAVE_PATH, STATE_LIST

DATA_PATH = SAVE_PATH / "psd/results"

for state in STATE_LIST:
    for elec in CHANNEL_NAMES:
        file_name = "da_bin_{}_{}_{}_{:.2f}.mat".format(state, elec, WINDOW, OVERLAP)
        data = loadmat(DATA_PATH / file_name)["score"]
        n, m = data.shape
        for i in range(n):
            for j in range(i + 1, m):
                data[j, i] = data[i, j]
        fig, ax = plt.subplots(figsize=(15, 15))
        fig.suptitle(elec, fontsize=20)
        # mat = ax.matshow(data, vmin=.5, vmax=.65, interpolation=None)
        mat = ax.matshow(data[:50, :50], vmin=.5, vmax=.65, interpolation=None)

        ax.set_xticklabels(FBIN_LIST, rotation=45)
        ax.set_yticklabels(FBIN_LIST)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        co = fig.colorbar(mat)
        co.set_ticks([.5, .525, .55, .575, .6, .625, .65])
        co.set_label("Decoding Accuracies")

        plt.savefig(
            "figures/reduced_da_bands_{}_{}_{}_{}.png".format(
                state, elec, WINDOW, OVERLAP
            ),
            dpi=150,
        )
        plt.close("all")
        # plt.show()
