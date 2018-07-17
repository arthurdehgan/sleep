import operator
from scipy.io import loadmat
import matplotlib.pyplot as plt
from params import SAVE_PATH, STATE_LIST, CHANNEL_NAMES

BANDS = ['delta ', 'theta ', 'alpha ', 'sigma ', 'beta  ']
# BANDS = ['delta ', 'theta ', 'alpha ', 'sigma ', 'beta  ', 'gamma1', 'gamma2']
COLORS = ['#DC9656', '#D8D8D8', '#86C1B9', '#BA8BAF', '#7CAFC2', '#A1B56C', '#AB4642']
SAVE_PATH /= 'psd/results'

for state in STATE_LIST:
    for elec in CHANNEL_NAMES:
        file_name = 'EFS_NoGamma_{}_{}_1000_0.00.mat'.format(state, elec)

        feats = loadmat(SAVE_PATH / file_name)['freqs'].ravel()
        feats = [elem.tolist() for elem in feats]
        flattened = []
        for elem in feats:
            flattened += elem
        counts = {}
        color = {}
        # most_comb = {}
        for i, band in enumerate(BANDS):
            color[band] = COLORS[i]
            counts[band] = flattened.count(band)
            # most_comb[band] = 0
        # most_viewed_feat = max(counts.items(), key=operator.itemgetter(1))[0]
        # for comb in feats:
            # if most_viewed_feat in comb:
                # if len(comb) == 1:
                    # most_comb[most_viewed_feat] -= 1
                # for band in comb:
                    # most_comb[band] += 1
        # print(state, elec, most_viewed_feat, ':', most_comb)
        prev = 0
        k = 0
        # for band in most_comb:
        for band in counts:
            # height = most_comb[band]
            height = counts[band]
            plt.bar(k, height, .95, color=color[band])
            # plt.bar(k, height, .95, bottom=prev, color=color[band])
            # prev = height
            k += 1
        # plt.ylim(0, 100)
        plt.title(state + ' ' + elec)
        plt.xticks(list(range(len(BANDS))), BANDS)
        plt.legend(BANDS)
        plt.savefig('figures/feature_selection_{}_{}_NoGamma.png'.format(state, elec))
        plt.close()
