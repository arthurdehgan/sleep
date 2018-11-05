"""Generate barplot and saves it."""
from math import ceil
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.io import loadmat
from scipy.stats import sem
from params import FREQ_DICT, STATE_LIST, SAVE_PATH, WINDOW, OVERLAP

# Use for binomial threshold (if no perm test has been done) :
from scipy.stats import binom


# Path where the figure will be saved
FIG_PATH = SAVE_PATH.dirname() / "figures"
# Path where the results are loaded from
# NAME_COSP = "moy_cosp"
# NAME_COV = "moy_cov"
# NAME_COSP = "cosp"
# NAME_COV = "cov"
# PREFIX = "classif_"
NAME_COSP = "subsamp_cosp"
NAME_COV = "subsamp_cov"
PREFIX = "bootstrapped_classif_"
MOY = "moy" in NAME_COSP
SUBSAMP = "subsamp" in NAME_COSP.split("_")
COSP_PATH = SAVE_PATH / NAME_COSP / "results/"
COV_PATH = SAVE_PATH / NAME_COV / "results"
PERM = "perm" in PREFIX.split("_")
PVAL = 0.001
if "Gamma1" in FREQ_DICT:
    del FREQ_DICT["Gamma1"]

MINMAX = [40, 80]
Y_LABEL = "Decoding accuracies (%)"
COLORS = ["#C2C2C2"] + list(sns.color_palette("deep"))

# COLORS = ['#DC9656', '#D8D8D8', '#86C1B9', '#BA8BAF',
# '#7CAFC2', '#A1B56C', '#AB4642']
WIDTH = .90
GRAPH_TITLE = ""
# GRAPH_TITLE = "Riemannian classifications"

RESOLUTION = 300
func_dict = {"sem": sem, "std": np.std}


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
def visualisation(pval, scoring, print_sem=None):
    sem_suffix = ""
    # states_suffix = "_AllStates"
    # gamma_suffix = ""
    labels = list(FREQ_DICT.keys())
    labels = ["Covariance"] + labels
    # if not gamma1:
    #     labels.remove("Gamma1")
    #     gamma_suffix = "_NoGamma2"
    # if not gamma2:
    #     labels.remove("Gamma2")
    #     gamma_suffix = "_NoGamma"
    # if not all_states:
    #     groups = STATE_LIST[2:]
    #     states_suffix = ""
    # else:
    # groups = STATE_LIST
    groups = STATE_LIST
    if print_sem is None:
        sem_suffix = "_NoSTD"
    # if not gamma:
    #     labels.remove("Gamma1")
    #     gamma_suffix = "_NoGamma"

    if SUBSAMP:
        metric = scoring
    else:
        metric = "data" if scoring == "acc" else "auc"

    nb_labels = len(labels)
    dat, sems = [], []
    thresholds = []
    for state in groups:
        temp_sem, temp, temp_thresh = [], [], []
        for lab in labels:
            if lab == "Covariance":
                file_name = COV_PATH / PREFIX + f"{NAME_COV}_{state}.mat"
            else:
                file_name = (
                    COSP_PATH / PREFIX
                    + f"{NAME_COSP}_{state}_{lab}_{WINDOW}_{OVERLAP:.2f}.mat"
                )

            try:
                print(file_name)
                data = loadmat(file_name)
                if PERM:
                    pscores = data["acc_pscores"][0]
                    ind = int(PVAL * len(pscores))
                    threshold = sorted(pscores)[-ind]
                if metric in list(data.keys()):
                    data = data[metric][0]
                else:
                    data = data["acc_score"][0]
                if data[0] < 1:
                    data *= 100
            except IOError:
                print(file_name, "not found.")
            except KeyError:
                print(file_name, metric, "key error")
            temp.append(np.mean(data))
            temp_sem.append(func_dict[print_sem](data))
            if PERM:
                if threshold < 1:
                    threshold *= 100
                temp_thresh.append(threshold)
        dat.append(temp)
        sems.append(temp_sem)
        if PERM:
            thresholds.append(temp_thresh)

    info_data = pd.read_csv(SAVE_PATH / "info_data.csv")[STATE_LIST]
    if SUBSAMP:
        # n_trials = 36 * int(input('Number of trials per subject ? '))
        n_trials = 36 * info_data.min().min()
        print(info_data.min().min())
        trials = [n_trials] * len(groups)
    elif MOY:
        n_trials = 36
        trials = [n_trials] * len(groups)
    else:
        trials = list(info_data.iloc[-1])
    if not PERM:
        thresholds = [
            100 * binom.isf(pval, n_trials, .5) / n_trials for n_trials in trials
        ]
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
        sem_val = sems[group]
        for i, val in enumerate(data):
            if PERM:
                t = thresholds[group][i]
            pos = i + 1
            if i == 1:
                temp += offset  # offset for the first bar
            color = COLORS[i]
            if print_sem:
                bars.append(
                    ax.bar(temp + pos, val, WIDTH, color=color, yerr=sem_val[i])
                )
            else:
                bars.append(ax.bar(temp + pos, val, WIDTH, color=color))
            start = (
                (temp + pos * WIDTH) / 2 + 1 - WIDTH
                if pos == 1 and temp == 0
                else temp + pos - len(data) / (2 * len(data) + 1)
            )
            end = start + WIDTH
            ax.plot([start, end], [t, t], "k--")
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

    file_name = PREFIX + f"{NAME_COSP}_{scoring}_{pval}_1000_0{sem_suffix}_NoGamma.png"
    print(FIG_PATH / file_name)
    save_path = str(FIG_PATH / file_name)
    fig.savefig(save_path, dpi=RESOLUTION)
    plt.close()


if __name__ == "__main__":
    """
    pvalues = [0.05, 0.01, 0.001]
    scores = ['auc', 'acc']
    options = [True, False]
    for pval in pvalues:
        for scoring in scores:
            for gamma in options:
                for sem in options:
                    for states in options:
                        visualisation(pval, scoring, gamma, sem, states)
    """
    visualisation(PVAL, "acc", "sem")
