'''Generate topomaps'''
from mne.viz import plot_topomap
from scipy.io import loadmat
from params import SAVE_PATH, STATE_LIST, CHANNEL_NAMES
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

DATA_PATH = SAVE_PATH / 'psd'
RESULTS_PATH = DATA_PATH / 'results'
POS_FILE = SAVE_PATH / '../Coord_EEG_1020.mat'
SENSORS_POS = loadmat(POS_FILE)['Cor']
# FREQS = ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta', 'Gamma1', 'Gamma2']
WINDOW = 1000
OVERLAP = 0
p = .001

for stage in STATE_LIST:

    fig = plt.figure(figsize=(5, 5))
    scores, pvalues = [], []
    for elec in CHANNEL_NAMES:
        file_name = 'perm_PSDM_{}_{}_{}_{:.2f}.mat'.format(
                    stage, elec, WINDOW, OVERLAP)
        try:
            score = loadmat(RESULTS_PATH / file_name)['score'].ravel()
        except TypeError:
            print(file_name)
        scores.append(score[0]*100)

        pvalue = loadmat(RESULTS_PATH / file_name)['pvalue'].ravel()
        pvalues.append(pvalue[0])

    DA = np.asarray(scores)
    da_pvalues = np.asarray(pvalues)

    da_mask = np.full((len(CHANNEL_NAMES)), False, dtype=bool)
    da_mask[da_pvalues <= p] = True
    mask_params = dict(marker='*', markerfacecolor='white', markersize=9,
                       markeredgecolor='white')

    data = {'name': 'Decoding Accuracies p<{}'.format(p), 'cmap': 'viridis',
            'mask': da_mask, 'cbarlim': [50, 65], 'data': DA}

    ch_show = True
    ax, _ = plot_topomap(data['data'], SENSORS_POS, res=128,
                         cmap=data['cmap'], show=False,
                         vmin=data['cbarlim'][0],
                         vmax=data['cbarlim'][1],
                         names=CHANNEL_NAMES, show_names=ch_show,
                         mask=data['mask'], mask_params=mask_params,
                         contours=0)
    fig.colorbar(ax, shrink=.65)

    file_name = 'topomap_all_multifeature_{}_p{}'.format(stage, str(p)[2:])
    savename = SAVE_PATH / '../figures' / file_name
    plt.savefig(savename, dpi=200)
