"""Generate barplot and saves it."""
from math import ceil
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import binom
from utils import super_count
from params import (
    STATE_LIST,
    SAVE_PATH,
    WINDOW,
    OVERLAP,
    REGIONS,
    CHANNEL_NAMES,
    FREQ_DICT,
)


FIG_PATH = SAVE_PATH.dirname() / "figures"
NAME = "EFS_NoGamma"
RESULT_PATH = SAVE_PATH / "psd/results/"

MINMAX = [40, 80]
Y_LABEL = "Decoding accuracies (%)"
COLORS = list(sns.color_palette("deep"))
WIDTH = .90
GRAPH_TITLE = "multifeature classification"

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


def get_max_key(dico):
    """Returns key with max value"""
    our_max = 0
    argmax = None
    for key, val in dico.items():
        if val > our_max:
            argmax = key
            our_max = val
    return argmax


# barplot parameters
def visualisation():
    labels = list(REGIONS)
    groups = STATE_LIST

    nb_labels = len(labels)
    score_stade = {}
    for stage in STATE_LIST:
        scores_regions = dict.fromkeys(REGIONS.keys(), 0)
        for region, elec_list in REGIONS.items():
            counts, all_count = {}, {}
            scores_elecs, freqs_elecs = {}, {}
            for elec in elec_list:
                file_name = "EFS_NoGamma_{}_{}_1000_0.00.mat".format(stage, elec)
                data = loadmat(RESULT_PATH / file_name)
                scores = data["test_scores"].ravel() * 100
                if stage == "S2" and elec == "F4":
                    all_count["Sigma"] = all_count.get("Sigma", 0) + 324
                    freqs_elecs[elec] = ["Sigma"] * 324
                    scores_elecs[elec] = scores
                    continue
                freqs = data["freqs"].ravel()
                freqs_stripped = [
                    freq.strip().capitalize() for sub in freqs for freq in sub
                ]
                freqs_elecs[elec] = freqs_stripped
                scores_elecs[elec] = [
                    scores[i] for i, sub in enumerate(freqs) for freq in sub
                ]
                count = super_count(freqs_stripped)
                counts[elec] = count
                for freq in FREQ_DICT:
                    all_count[freq] = all_count.get(freq, 0) + count.get(freq, 0)

            freq = get_max_key(all_count)
            for elec in elec_list:
                best_freq_index = np.where(np.asarray(freqs_elecs[elec]) == freq)[0]
                scores_regions[region] += np.mean(
                    np.asarray(scores_elecs[elec])[best_freq_index]
                ) / len(elec_list)

        score_stade[stage] = scores_regions

    fig = plt.figure(figsize=(10, 5))  # size of the figure

    # Generating the barplot (do not change)
    ax = plt.axes()
    temp = 0
    info_data = pd.read_csv(SAVE_PATH / "info_data.csv")[STATE_LIST]
    trials = list(info_data.iloc[-1])
    thresholds = [
        100 * binom.isf(0.001, n_trials, .5) / n_trials for n_trials in trials
    ]
    for j, group in enumerate(groups):
        bars = []
        for i, region in enumerate(REGIONS):
            data = score_stade[group][region]
            pos = i + 1
            color = COLORS[i]
            bars.append(ax.bar(temp + pos, data, WIDTH, color=color))
        temp += pos + 1
        start = j * (pos + 1) + .5
        end = start + len(REGIONS)
        ax.plot([start, end], [thresholds[j], thresholds[j]], "k--", label="p=0.001")

    ax.set_ylabel(Y_LABEL)
    ax.set_ylim(bottom=MINMAX[0], top=MINMAX[1])
    ax.set_title(GRAPH_TITLE)
    ax.set_xticklabels(groups)
    ax.set_xticks(
        [ceil(nb_labels / 2) + i * (1 + nb_labels) for i in range(len(groups))]
    )

    plt.legend(bars, labels, fancybox=False, shadow=False)

    file_name = f"{NAME}_1000_0.png"
    print(FIG_PATH / file_name)
    save_path = str(FIG_PATH / file_name)
    fig.savefig(save_path, dpi=RESOLUTION)


if __name__ == "__main__":
    visualisation()
