"""Generate barplot and saves it."""
from math import ceil
import matplotlib.pyplot as plt
from path import Path as path

# Use for binomial threshold (if no perm test has been done) :
# from scipy.stats import binom


def autolabel(rects, thresh):
    """Attach a text label above each bar displaying its height."""
    for rect in rects:
        height = rect.get_height()
        if height > thresh:
            color = "green"
        else:
            color = "black"
        if height != 0:
            ax.text(
                rect.get_x() + rect.get_width() / 2.,
                1. * height,
                "%d" % int(height),
                ha="center",
                va="bottom",
                color=color,
            )


# Path where the figure will be saved
SAVE_PATH = path("/home/arthur/tests")

# barplot parameters
labels = ["data{}".format(i) for i in range(7)]  # label for each bar
nb_labels = len(labels)
GROUPS = ["group{}".format(i) for i in range(7)]  # label for each GROUPS
data = [50, 60, 50, 65, 80, 40, 50]  # the actual data
thresholds = [50] * 7  # The thresholds
MINMAX = [30, 90]  # Minimum and maximum of x axis scale
Y_LABEL = "Decoding accuracies"  # legend for the y axis

COLORS = [
    "#DC9656",
    "#D8D8D8",
    "#86C1B9",
    "#BA8BAF",
    "#7CAFC2",
    "#A1B56C",
    "#AB4642",
]  # hex code of bar COLORS '#F7CA88'
WIDTH = .90  # WIDTH of the bars, change at your own risks, it might break
GRAPH_TITLE = "Titre du barplot"  # Graph title
FILE_NAME = "nom_fichier.png"  # File name when saved
RESOLUTION = 300  # resolution of the saved image in pixel per inch
fig = plt.figure(figsize=(10, 5))  # size of the figure

# Generating the barplot (do not change)
ax = plt.axes()
temp = 0
for group in range(len(GROUPS)):
    bars = []
    for i, val in enumerate(data):
        pos = i + 1
        t = thresholds[i]
        bars.append(ax.bar(temp + pos, val, WIDTH, color=COLORS[i]))
        start = (
            (temp + pos * WIDTH) / 2 + 1 - WIDTH
            if pos == 1 and temp == 0
            else temp + pos - len(data) / (2 * len(data) + 1)
        )
        end = start + WIDTH
        ax.plot([start, end], [t, t], "k--")
        autolabel(bars[i], t)
    temp += pos + 1

ax.set_ylabel(Y_LABEL)
ax.set_ylim(bottom=MINMAX[0], top=MINMAX[1])
ax.set_title(GRAPH_TITLE)
ax.set_xticklabels(GROUPS)
ax.set_xticks([ceil(nb_labels / 2) + i * (1 + nb_labels) for i in range(len(GROUPS))])
ax.legend(
    bars,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.05),
    fancybox=True,
    shadow=True,
    ncol=len(labels),
)

print(FILE_NAME)
plt.savefig(FILE_NAME, dpi=RESOLUTION)
# plt.show()
