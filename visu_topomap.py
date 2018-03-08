'''Generate topomaps'''
from mne.viz import plot_topomap
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from scipy.stats import zscore
from params import SAVE_PATH, STATE_LIST, N_ELEC, CHANNEL_NAMES

DATA_PATH = SAVE_PATH / 'psd'
RESULTS_PATH = DATA_PATH / 'results'
POS_FILE = SAVE_PATH / 'Coord_EEG_1020.mat'
SENSORS_POS = loadmat(POS_FILE)['Cor']
FREQS = ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta', 'Gamma1', 'Gamma2']
WINDOW = 1000
OVERLAP = 0
p = 1/1001

for stage in STATE_LIST:

    k, j = 1, 1
    fig = plt.figure(figsize=(16, 18))
    for freq in FREQS:

        scores, pvalues = [], []
        dreamer, ndreamer = [], []
        for elec in range(N_ELEC):
            file_name = 'perm_PSD_{}_{}_{}_{}_{:.2f}.mat'.format(
                        stage, freq, elec, WINDOW, OVERLAP)
            score = loadmat(RESULTS_PATH / file_name)['score'].ravel()
            scores.append(score[0]*100)

            pvalue = loadmat(RESULTS_PATH / file_name)['pvalue'].ravel()
            pvalues.append(pvalue[0])

            file_name = 'PSD_{}_{}_{}_{}_{:.2f}.mat'.format(
                        stage, freq, elec, WINDOW, OVERLAP)
            PSD = loadmat(DATA_PATH / file_name)['data'].ravel()
            ndreamer.append(np.mean([e.ravel().mean() for e in PSD[18:]]))
            dreamer.append(np.mean([e.ravel().mean() for e in PSD[:18]]))

        ttest = loadmat(RESULTS_PATH / 'ttest_perm_{}_{}.mat'.format(stage, freq))
        tt_pvalues_r = ttest['p_right'][0]
        tt_pvalues_l = ttest['p_left'][0]
        t_values = zscore(ttest['t_values'][0])
        dreamer = np.asarray(dreamer)
        ndreamer = np.asarray(ndreamer)
        DA = np.asarray(scores)
        da_pvalues = np.asarray(pvalues)
        RPC = zscore((dreamer - ndreamer) / ndreamer)
        dreamer = zscore(dreamer)
        ndreamer = zscore(ndreamer)

        da_mask = np.full((len(CHANNEL_NAMES)), False, dtype=bool)
        tt_mask = np.full((len(CHANNEL_NAMES)), False, dtype=bool)
        tt_mask[tt_pvalues_r <= 0.0005] = True
        tt_mask[tt_pvalues_l <= 0.0005] = True
        da_mask[da_pvalues <= p] = True
        mask_params = dict(marker='*', markerfacecolor='white', markersize=9,
                           markeredgecolor='white')

        data = [{'name': 'PSD dreamer', 'cmap': 'jet', 'mask': None,
                 'cbarlim': [min(dreamer), max(dreamer)], 'data': dreamer},
                {'name': 'PSD non-dreamer', 'cmap': 'jet', 'mask': None,
                 'cbarlim': [min(ndreamer), max(ndreamer)], 'data': ndreamer},
                {'name': 'Relative Power Changes', 'cmap': 'inferno',
                 'mask': None, 'cbarlim': [min(RPC), max(RPC)], 'data': RPC},
                {'name': 'ttest permutations p<0.001', 'data': t_values,
                 'cmap': 'viridis', 'mask': tt_mask,
                 'cbarlim': [min(t_values), max(t_values)]},
                {'name': 'Decoding Accuracies', 'cmap': 'viridis',
                 'mask': da_mask, 'cbarlim': [50, 65], 'data': DA}]

        for i, subset in enumerate(data):
            plt.subplot(len(FREQS), len(data), i+k)
            ch_show = False if i > 1 else True
            ax, _ = plot_topomap(subset['data'], SENSORS_POS, res=128,
                                 cmap=subset['cmap'], show=False,
                                 vmin=subset['cbarlim'][0],
                                 vmax=subset['cbarlim'][1],
                                 names=CHANNEL_NAMES, show_names=ch_show,
                                 mask=subset['mask'], mask_params=mask_params,
                                 contours=0)
            fig.colorbar(ax, shrink=.65)

        fig.text(0.004, 1 - (2*j-1)/14, freq, va='center', rotation=90, size='x-large')
        j += 1
        k += 5

    for i, subset in enumerate(data):
        fig.text(i/5 + .08, .008, subset['name'], ha='center', size='x-large')
    plt.tight_layout()
    # plt.suptitle('Topomap {}'.format(stage))
    file_name = 'topomap_{}'.format(stage)
    plt.savefig(SAVE_PATH / 'figures' / file_name, dpi=300)
