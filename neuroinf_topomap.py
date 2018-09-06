'''Generate topomaps'''
from mne.viz import plot_topomap
from scipy.io import loadmat
from params import SAVE_PATH, CHANNEL_NAMES
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

DATA_PATH = SAVE_PATH / 'psd'
TTEST_RESULTS_PATH = DATA_PATH / 'results'
solver = 'svd'
RESULTS_PATH = DATA_PATH / 'results/'
POS_FILE = SAVE_PATH / '../Coord_EEG_1020.mat'
SENSORS_POS = loadmat(POS_FILE)['Cor']
FREQS = ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta', 'Gamma1']
STATE_LIST = ['S2', 'NREM', 'Rem']
# prefix = 'bootstrapped_perm_subsamp_'
prefix = 'perm_'
WINDOW = 1000
OVERLAP = 0
p = .01

fig = plt.figure(figsize=(15, 8))
j = 1
for stage in STATE_LIST:
    for freq in FREQS:
        plt.subplot(len(STATE_LIST), len(FREQS), j)
        scores, pvalues = [], []
        for elec in CHANNEL_NAMES:
            file_name = prefix + 'PSD_{}_{}_{}_{}_{:.2f}.mat'.format(
                        stage, freq, elec, WINDOW, OVERLAP)
            try:
                score = loadmat(RESULTS_PATH / file_name)
                pvalue = score['pvalue'].ravel()
                score = score['score'].ravel().mean()
            except TypeError:
                score = [.5]
                pvalue = [1]
                print(RESULTS_PATH / file_name)
            scores.append(score*100)
            pvalues.append(pvalue[0])

        DA = np.asarray(scores)
        da_pvalues = np.asarray(pvalues)

        da_mask = np.full((len(CHANNEL_NAMES)), False, dtype=bool)
        da_mask[da_pvalues <= p] = True
        mask_params = dict(marker='*', markerfacecolor='white', markersize=9,
                           markeredgecolor='white')

        subset = {'name': 'Decoding Accuracies p<{}'.format(p), 'cmap': 'viridis',
                 'mask': da_mask, 'cbarlim': [50, 65], 'data': DA}

        ch_show = False if j > 1 else True
        ax, _ = plot_topomap(subset['data'], SENSORS_POS, res=128,
                             cmap=subset['cmap'], show=False,
                             vmin=subset['cbarlim'][0],
                             vmax=subset['cbarlim'][1],
                             names=CHANNEL_NAMES, show_names=ch_show,
                             mask=subset['mask'], mask_params=mask_params,
                             contours=0)

        j += 1

fig.colorbar(ax)
plt.subplots_adjust(left=None, bottom=0.05, right=None, top=None,
                    wspace=None, hspace=None)
plt.tight_layout()
file_name = 'topomap_neuroinf_p{}'.format(str(p)[2:])
plt.savefig(SAVE_PATH / '../figures' / file_name, dpi=600)
