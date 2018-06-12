'''Ttest for PSD values.
by Arthur Dehgan'''

from scipy.io import loadmat, savemat
import numpy as np
from joblib import Parallel, delayed
from ttest_perm_indep import ttest_perm_unpaired
from params import STATE_LIST, FREQ_DICT, SAVE_PATH, WINDOW,\
                   OVERLAP, CHANNEL_NAMES

SAVE_PATH = SAVE_PATH / 'psd'
RESULT_PATH = SAVE_PATH / 'results'
n_perm = 9999


def main(stade, freq):
    dreamers, ndreamers = [], []
    for elec in CHANNEL_NAMES:

        file_path = SAVE_PATH / 'PSD_{}_{}_{}_{}_{:.2f}.mat'.format(
                                 stade, freq, elec, WINDOW, OVERLAP
                                 )
        try:
            X = loadmat(file_path)['data'].ravel()
        except KeyError:
            print(file_path, 'corrupted')
        except IOError:
            print(file_path, 'Not Found')
        dreamer = X[:18]
        ndreamer = X[18:]
        for i in range(len(dreamer)):
            dreamer[i] = dreamer[i].mean()
            ndreamer[i] = ndreamer[i].mean()

        dreamers.append(dreamer)
        ndreamers.append(ndreamer)

    dreamers = np.asarray(dreamers, dtype=float).T
    ndreamers = np.asarray(ndreamers, dtype=float).T
    tval, pvalues = ttest_perm_unpaired(cond1=dreamers,
                                        cond2=ndreamers,
                                        n_perm=n_perm,
                                        equal_var=False,
                                        two_tailed=True,
                                        n_jobs=-2)

    data = {'p_values': np.asarray(pvalues),
            't_values': tval}
    file_path = RESULT_PATH / 'ttest_perm_{}_{}.mat'.format(stade, freq)
    savemat(file_path, data)


if __name__ == '__main__':
    Parallel(n_jobs=1)(delayed(main)(stade, freq)
                       for stade in STATE_LIST for freq in FREQ_DICT)
