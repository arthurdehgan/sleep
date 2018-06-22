from utils import StratifiedLeave2GroupsOut, create_groups
from scipy.io import savemat, loadmat
import numpy as np
# from numpy.random import permutation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score
from params import CHANNEL_NAMES, LABEL_PATH, SUBJECT_LIST, SAVE_PATH, path

N_FBIN = 45
WINDOW = 1000
OVERLAP = 0
N_PERMUTATIONS = 1000
SLEEP_LIST = ['S1', 'S2', 'SWS', 'Rem', 'NREM', 'AWA']
DATA_PATH = SAVE_PATH / 'psd'
SAVE_PATH = DATA_PATH / 'results'
SUB_LIST = ['s' + str(e) for e in SUBJECT_LIST]


if __name__ == '__main__':
    for state in SLEEP_LIST:
        print(state)
        for elec in CHANNEL_NAMES:
            results_file_path = SAVE_PATH / \
                'da_bin_{}_{}_{}_{:.2f}.mat'.format(
                    state, elec, WINDOW, OVERLAP)

            if not results_file_path.exists():
                y = loadmat(LABEL_PATH / state + '_labels.mat')['y'].ravel()
                y, groups = create_groups(y)

                dataset = []
                for sub in SUB_LIST:
                    data_file_path = DATA_PATH / \
                            'PSDs_{}_{}_{}_{}_{:.2f}.mat'.format(
                                state, sub, elec, WINDOW, OVERLAP)
                    if path(data_file_path).isfile():
                        dataset.append(loadmat(data_file_path)['data'])
                    else:
                        print(path(data_file_path) + ' Not found')
                dataset = np.vstack(dataset)

                scores = np.ones((N_FBIN, N_FBIN))*.5
                for fbin_min in range(N_FBIN):
                    for fbin_max in range(fbin_min + 1, N_FBIN):
                        X = np.mean(
                            dataset[:, fbin_min:fbin_max],
                            axis=1).reshape(-1, 1)
                        sl2go = StratifiedLeave2GroupsOut()
                        clf = LDA()
                        perm_scores = []
                        pvalue = 0
                        good_scores = cross_val_score(cv=sl2go, estimator=clf,
                                                      X=X, y=y,
                                                      groups=groups, n_jobs=1)
                        scores[fbin_min, fbin_max] = good_scores.mean()

                save = {'score': scores}
                savemat(results_file_path, save)
