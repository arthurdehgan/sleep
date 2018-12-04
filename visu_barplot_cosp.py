"""Generate barplot and saves it."""
from math import ceil
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.io import loadmat
from params import FREQ_DICT, STATE_LIST, SAVE_PATH, WINDOW, OVERLAP


FIG_PATH = SAVE_PATH.dirname() / "figures"
# NAME_COSP = "moy_cosp"
# NAME_COV = "moy_cov"
# NAME_COSP = "cosp"
# NAME_COV = "cov"
# PREFIX = "classif_"
NAME_COSP = "cosp"
NAME_COV = "cov"
PREFIX = "bootstrapped_subsamp_"
MOY = "moy" in NAME_COSP
SUBSAMP = "subsamp" in NAME_COSP.split("_")
COSP_PATH = SAVE_PATH / NAME_COSP / "results/"
COV_PATH = SAVE_PATH / NAME_COV / "results"
PERM = True
PVAL = 0.001

MINMAX = [40, 80]
Y_LABEL = "Decoding accuracies (%)"
COLORS = ["#C2C2C2"] + list(sns.color_palette("deep"))
WIDTH = .90
GRAPH_TITLE = "Riemannian classifications"

RESOLUTION = 300


def autolabel(ax, rects, thresh):
    """Attach a text label above each bar displaying its height."""
    for rect in rects:
        height = rect.get_height()
        width = rect.get_width()
        if height > thresh:
            color = "green"
        else:
            color = "black"

        if height != 0:
            ax.text(
                rect.get_x() + width / 2.,
                width + 1. * height,
                "%d" % int(height),
                ha="center",
                va="bottom",
                color=color,
                size=14,
            )
    return ax


# barplot parameters
def visualisation(pval):
    scoring = "acc"
    labels = list(FREQ_DICT.keys())
    labels = ["Covariance"] + labels
    groups = STATE_LIST

    nb_labels = len(labels)
    dat, stds = [], []
    thresholds = []
    for state in groups:
        temp_std, temp, temp_thresh = [], [], []
        for lab in labels:
            if lab == "Covariance":
                file_name = COV_PATH / f"{PREFIX}{NAME_COV}_{state}.mat"
                perm_fname = COV_PATH / f"perm_{NAME_COV}_{state}.mat"
            else:
                file_name = (
                    COSP_PATH
                    / f"{PREFIX}{NAME_COSP}_{state}_{lab}_{WINDOW}_{OVERLAP:.2f}.mat"
                )
                perm_fname = (
                    COSP_PATH
                    / f"perm_{NAME_COSP}_{state}_{lab}_{WINDOW}_{OVERLAP:.2f}.mat"
                )

            try:
                data = loadmat(file_name)
                n_rep = int(data["n_rep"])
                data = np.asarray(data[scoring][0]) * 100
                n_cv = int(len(data) / n_rep)
                if PERM:
                    data_perm = loadmat(perm_fname)
                    pscores = np.asarray(data_perm["acc_pscores"][0]) * 100
                    ind = int(PVAL * len(pscores))
                    threshold = sorted(pscores)[-ind]
            except IOError:
                print(file_name, "not found.")
            except KeyError:
                print(file_name, "key error")

            temp.append(np.mean(data))
            std_value = np.std(
                [np.mean(data[i * n_cv : (i + 1) * n_cv]) for i in range(n_rep)]
            )
            temp_std.append(std_value)
            if PERM:
                if threshold < 1:
                    threshold *= 100
                temp_thresh.append(threshold)
        dat.append(temp)
        stds.append(temp_std)
        if PERM:
            thresholds.append(temp_thresh)

    fig = plt.figure(figsize=(10, 5))  # size of the figure

    # Generating the barplot (do not change)
    ax = plt.axes()
    temp = 0
    offset = .4
    for group in range(len(groups)):
        bars = []
        if not PERM:
            t = thresholds[group]
        data = dat[group]
        std_val = stds[group]
        for i, val in enumerate(data):
            if PERM:
                t = thresholds[group][i]
            pos = i + 1
            if i == 1:
                temp += offset  # offset for the first bar
            color = COLORS[i]
            bars.append(ax.bar(temp + pos, val, WIDTH, color=color, yerr=std_val[i]))
            start = (
                (temp + pos * WIDTH) / 2 + 1 - WIDTH
                if pos == 1 and temp == 0
                else temp + pos - len(data) / (2 * len(data) + 1)
            )
            end = start + WIDTH
            ax.plot([start, end], [t, t], "k--", label="p < {}".format(PVAL))
            # ax = autolabel(ax, bars[i], t)

        temp += pos + 1

    ax.set_ylabel(Y_LABEL)
    ax.set_ylim(bottom=MINMAX[0], top=MINMAX[1])
    ax.set_title(GRAPH_TITLE)
    ax.set_xticklabels(groups)
    ax.set_xticks(
        [
            ceil(nb_labels / 2) + offset + i * (1 + offset + nb_labels)
            for i in range(len(groups))
        ]
    )
    # labels[-1] = labels[-1][:-1]
    labels = ["Covariance"] + [elem + " cospec" for elem in FREQ_DICT]
    # ax.legend(bars, labels, frameon=False)
    ax.legend(
        bars,
        labels,
        # loc="upper center",
        # bbox_to_anchor=(0.5, -0.05),
        fancybox=False,
        shadow=False,
        # ncol=len(labels),
    )

    file_name = PREFIX + f"{NAME_COSP}_{scoring}_{pval}_1000_0.png"
    print(FIG_PATH / file_name)
    save_path = str(FIG_PATH / file_name)
    fig.savefig(save_path, dpi=RESOLUTION)
    plt.close()


if __name__ == "__main__":
    visualisation(PVAL)
