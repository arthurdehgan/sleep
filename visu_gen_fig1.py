import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.io import loadmat
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from params import STATE_LIST, SAVE_PATH, CHANNEL_NAMES, SUBJECT_LIST

FIG_PATH = SAVE_PATH.parent / "figures"
COSP_PATH = SAVE_PATH / "cosp"
COV_PATH = SAVE_PATH / "cov"
PSD_PATH = SAVE_PATH / "psd/per_bin"

HR_LABELS = ("HR", "HR mean")
LR_LABELS = ("LR", "LR mean")

STATE = "S2"
FREQ = "Delta"
FONTSIZE = 16


def prepare_recallers(data):
    HR = data[:18]
    LR = data[18:]
    for i, submat in enumerate(HR):
        if i == 9:
            vals = []
            for mat in submat:
                vals.append(mat[-3, -3])
            std = np.std(vals)
            for j, val in enumerate(vals):
                if val > 2 * std:
                    np.delete(submat, j, 0)
        HR[i] = submat.mean(axis=0)
    for i, submat in enumerate(LR):
        LR[i] = submat.mean(axis=0)
    HR = np.delete(HR, 9, 0)  # subject 10 has artifacts on FC2
    HR = HR.mean()
    HR /= HR.max()
    LR = LR.mean()
    LR /= LR.max()
    return np.flip(HR, 0), np.flip(LR, 0)


def compute(val, k):
    return math.log(val / (k + 1))


def do_matrix(fig, mat):
    mat = fig.pcolormesh(mat, vmin=0, vmax=1)
    fig.set_xticklabels(CHANNEL_NAMES, rotation=90)
    fig.set_yticklabels(reversed(CHANNEL_NAMES))
    ticks = [i + .5 for i, _ in enumerate(CHANNEL_NAMES)]
    fig.set_xticks(ticks)
    fig.set_yticks(ticks)
    fig.tick_params(labeltop=False, labelbottom=True, top=False)
    return mat


COV_NAME = COV_PATH / "cov_{}.mat".format(STATE)
DATA = loadmat(COV_NAME)["data"].ravel()
HR_COV, LR_COV = prepare_recallers(DATA)

cosp_name = COSP_PATH / "cosp_{}_{}_1000_0.00.mat".format(STATE, FREQ)
data = loadmat(cosp_name)
data = data["data"].ravel()
HR_COSP, LR_COSP = prepare_recallers(data)

all_subs = []
for sub in SUBJECT_LIST:
    all_elecs = []
    for elec in CHANNEL_NAMES:
        data = []
        for state in STATE_LIST:
            psd_name = PSD_PATH / "PSDs_{}_s{}_{}_1000_0.00.mat".format(
                state, sub, elec
            )
            data.append(loadmat(psd_name)["data"].mean(axis=0))
        all_states = np.asarray(data).mean(axis=0)
        all_elecs.append(all_states)
    all_subs.append(np.asarray(all_elecs).mean(axis=0))
all_subs = np.asarray(all_subs)
hdr = all_subs[18:]
hdr = [[compute(a, i) for a in dat] for i, dat in enumerate(hdr)]
ldr = all_subs[:18]
ldr = [[compute(a, i) for a in dat] for i, dat in enumerate(ldr)]


fig = plt.figure(figsize=(25, 10))
ax1 = plt.subplot2grid((2, 5), (0, 0), colspan=2, rowspan=2)
ax2 = plt.subplot2grid((2, 5), (0, 2))
ax3 = plt.subplot2grid((2, 5), (0, 3))
ax4 = plt.subplot2grid((2, 5), (1, 2))
ax5 = plt.subplot2grid((2, 5), (1, 3))
ax6 = plt.subplot2grid((2, 5), (1, 4))

for i in range(len(hdr)):
    ax1.plot(
        range(1, 46), ldr[i], color="skyblue", label="_nolegend_" if i > 0 else "LR"
    )
    ax1.plot(
        range(1, 46), hdr[i], color="peachpuff", label="_nolegend_" if i > 0 else "HR"
    )
ax1.plot(range(1, 46), np.mean(hdr, axis=0), color="red", label="HR mean")
ax1.plot(range(1, 46), np.mean(ldr, axis=0), color="blue", label="LR mean")
ax1.legend(fontsize=FONTSIZE - 2, frameon=False)
ax1.set_xlabel("Frequency (Hz)", fontsize=FONTSIZE - 2)
ax1.set_ylabel("Power Spectral Density (dB/Hz)", fontsize=FONTSIZE)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
do_matrix(ax2, LR_COV)
ax2.set_ylabel("Covariance", fontsize=FONTSIZE)
do_matrix(ax3, HR_COV)
do_matrix(ax4, LR_COSP)
ax4.set_ylabel("Cospectrum", fontsize=FONTSIZE)
ax4.set_xlabel("Low Recallers", fontsize=FONTSIZE)
mat = do_matrix(ax5, HR_COSP)
ax5.set_xlabel("High Recallers", fontsize=FONTSIZE)

TICKS = [0, .2, .4, .6, .8, 1]

fig.colorbar(mat, ax=ax6, ticks=TICKS, orientation="vertical")

plt.tight_layout(pad=1)
save_name = str(FIG_PATH / "Figure_1.png")
plt.savefig(save_name, dpi=300)
