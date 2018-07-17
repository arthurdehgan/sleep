"""Generate barplot and saves it."""
from math import ceil
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.stats import sem, binom
from params import FREQ_DICT, STATE_LIST, SAVE_PATH, WINDOW, OVERLAP
# Use for binomial threshold (if no perm test has been done) :


# Path where the figure will be saved
FIG_PATH = SAVE_PATH.dirname() / 'figures'
# Path where the results are loaded from
prefix = 'moy_'
COSP_PATH = SAVE_PATH / prefix + 'cosp/results'
COV_PATH = SAVE_PATH / 'cov/results'

MINMAX = [30, 100]
Y_LABEL = 'Decoding accuracies'
COLORS = ['#DC9656', '#D8D8D8', '#86C1B9', '#BA8BAF',
          '#7CAFC2', '#A1B56C', '#AB4642']
WIDTH = .90
GRAPH_TITLE = "results time covariance" +\
    " and cospectrum window=1000, overlap=0"

RESOLUTION = 300
N_TRIALS = 36


def autolabel(ax, rects, thresh):
    """Attach a text label above each bar displaying its height."""
    for rect in rects:
        height = rect.get_height()
        if height > thresh:
            color = 'green'
        else:
            color = 'black'
        if height != 0:
            ax.text(rect.get_x() + rect.get_width()/2., 1.*height,
                    '%d' % int(height),
                    ha='center', va='bottom', color=color)
    return ax


# barplot parameters
def visualisation(pval, metric, print_sem, all_states, gamma, BINOM=False):
    sem_suffix = ''
    states_suffix = '_AllStates'
    gamma_suffix = ''
    labels = list(FREQ_DICT.keys())
    labels = ['Cov'] + labels
    if not gamma:
        labels.remove('Gamma1')
        gamma_suffix = '_NoGamma'
    if not all_states:
        groups = STATE_LIST[2:]
        states_suffix = ''
    else:
        groups = STATE_LIST
    if not print_sem:
        sem_suffix = '_NoSEM'

    nb_labels = len(labels)
    dat = []
    sems = []
    thresholds = []
    for state in groups:
        temp = []
        temp_sem = []
        for lab in labels:
            if lab == 'Cov':
                file_name = COV_PATH / 'perm_' + prefix +\
                    'cov_{}.mat'.format(state)
            else:
                file_name = COSP_PATH / 'perm_' + prefix +\
                    'cosp_{}_{}_{}_{:.2f}.mat'.format(
                        state, lab, WINDOW, OVERLAP)
            try:
                mat_file = loadmat(file_name)
                data = mat_file[metric]
                pscores = sorted(mat_file[metric + '_pscores'].ravel())
            except IOError:
                print(file_name, 'not found.')
            temp.append(100*np.mean(data))
            temp_sem.append(sem(data.ravel()))
            thresholds.append(100*pscores[-int(pval * len(pscores) + 1)])
        dat.append(temp)
        sems.append(temp_sem)

    if BINOM:
        thresholds = [100*binom.isf(pval, N_TRIALS, .5)/N_TRIALS] * N_TRIALS

    fig = plt.figure(figsize=(10, 5))  # size of the figure

    # Generating the barplot (do not change)
    ax = plt.axes()
    temp = 0
    for group in range(len(groups)):
        bars = []
        data = dat[group]
        sem_val = sems[group]

        for i, val in enumerate(data):
            pos = i + 1
            t_ind = group if BINOM else i + group
            t = thresholds[t_ind]

            if print_sem:
                bars.append(ax.bar(
                    temp + pos, val, WIDTH,
                    color=COLORS[i], yerr=100*sem_val[i]))
            else:
                bars.append(ax.bar(temp + pos, val, WIDTH, color=COLORS[i]))

            start = (temp+pos*WIDTH)/2 + 1-WIDTH \
                if pos == 1 and temp == 0 \
                else temp+pos - len(data)/(2*len(data)+1)

            end = start + WIDTH
            ax.plot([start, end], [t, t], "k-")
            ax = autolabel(ax, bars[i], t)
        temp += pos+1

    ax.set_ylabel(Y_LABEL)
    ax.set_ylim(bottom=MINMAX[0], top=MINMAX[1])
    ax.set_title(GRAPH_TITLE)
    ax.set_xticklabels(groups)
    ax.set_xticks([ceil(nb_labels/2)+i*(1+nb_labels) for i in range(
        len(groups))])
    ax.legend(bars, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=len(labels))

    FILE_NAME = "full_trial_{}_{}_1000_0{}{}{}.png".format(
        metric, pval, sem_suffix, states_suffix, gamma_suffix)
    print(FIG_PATH / FILE_NAME)
    save_path = str(FIG_PATH / FILE_NAME)
    fig.savefig(save_path, dpi=RESOLUTION)
    plt.close()


if __name__ == '__main__':
    # for scor in ['auc', 'acc']:
    visualisation(.01, 'acc', True, True, True)
