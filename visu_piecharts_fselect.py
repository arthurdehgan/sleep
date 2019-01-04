"""Generate topomaps"""
from scipy.io import loadmat
import matplotlib.pyplot as plt
from utils import super_count
from params import SAVE_PATH, STATE_LIST, CHANNEL_NAMES, REGIONS

plt.switch_backend("agg")

DATA_PATH = SAVE_PATH / "psd"
RESULTS_PATH = DATA_PATH / "results/"
FREQS = ["Delta", "Theta", "Alpha", "Sigma", "Beta"]
GRID_SIZE = (6, 4)

plt.figure(figsize=(8, 10))
for j, stage in enumerate(STATE_LIST):
    counts, all_count = {}, {}
    for elec in CHANNEL_NAMES:
        file_name = "EFS_{}_{}_1000_0.00.mat".format(stage, elec)
        freqs = loadmat(RESULTS_PATH / file_name)["freqs"].ravel()
        count = super_count(
            [freq.strip().capitalize() for sub in freqs for freq in sub]
        )
        counts[elec] = count
        for freq in FREQS:
            all_count[freq] = all_count.get(freq, 0) + count.get(freq, 0)

    plt.subplot2grid(GRID_SIZE, (0, j))
    plt.pie([all_count[freq] for freq in FREQS])
    if j == 0:
        plt.ylabel("All Stages")
    plt.xlabel(stage, verticalalignment="top")

    i = 1
    for region in REGIONS:
        elecs = REGIONS[region]
        sub_count = {}
        for freq in FREQS:
            sub_count[freq] = sum([counts[elec].get(freq, 0) for elec in elecs])
        plt.subplot2grid(GRID_SIZE, (i, j))
        plt.pie([sub_count[freq] for freq in FREQS])
        plt.tight_layout()
        if j == 0:
            plt.ylabel(region)
        i += 1

FILE_NAME = "EFS_piechart"
print(file_name)
plt.legend(
    FREQS,
    loc="upper center",
    bbox_to_anchor=(-1.8, -0.05),
    fancybox=False,
    shadow=False,
    ncol=len(FREQS),
)
plt.tight_layout()
plt.savefig(SAVE_PATH.parent / "figures" / FILE_NAME, dpi=300)
