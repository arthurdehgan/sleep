'''Ttest for PSD values.
by Arthur Dehgan'''

from scipy.io import loadmat, savemat
import numpy as np
from joblib import Parallel, delayed
from ttest_perm_indep import ttest_perm_ind_maxcor
from params import STATE_LIST, FREQ_DICT, N_ELEC, SAVE_PATH, WINDOW, OVERLAP

DATA_PATH = SAVE_PATH / 'psd'
SAVE_PATH = DATA_PATH / 'results'
n_perm = 9999
p_val = 0.0005


def main(stade, freq):
    dreamers, ndreamers = [], []
    for elec in range(N_ELEC):

        file_path = DATA_PATH / 'PSD_{}_{}_{}_{}_{:.2f}'.format(
                                 stade, freq, elec, WINDOW, OVERLAP
                                 )
        try:
            X = loadmat(file_path)['data'].ravel()
        except KeyError:
            print(file_path, 'corrupted')
        dreamer = X[:18]
        ndreamer = X[18:]
        for i in range(len(dreamer)):
            dreamer[i] = dreamer[i].mean()
            ndreamer[i] = ndreamer[i].mean()

        dreamers.append(dreamer)
        ndreamers.append(ndreamer)

    dreamers = np.asarray(dreamers, dtype=float).T
    ndreamers = np.asarray(ndreamers, dtype=float).T
    tval, pvalues = ttest_perm_ind_maxcor(cond1=dreamers,
                                          cond2=ndreamers,
                                          n_perm=n_perm,
                                          equal_var=False,
                                          two_tailed=True)
    p_right = pvalues[0]
    p_left = pvalues[1]

    data = {'p_right': np.asarray(p_right),
            'p_left': np.asarray(p_left),
            't_values': tval}
    file_path = SAVE_PATH / 'ttest_perm_{}_{}.mat'.format(stade, freq)
    savemat(file_path, data)


if __name__ == '__main__':
    Parallel(n_jobs=-2)(delayed(main)(stade, freq)
                        for stade in STATE_LIST for freq in FREQ_DICT)
