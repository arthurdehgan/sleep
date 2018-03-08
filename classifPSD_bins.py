from utils import StratifiedLeave2GroupsOut, create_groups
from scipy.io import savemat, loadmat
import numpy as np
from numpy.random import permutation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score
from params import n_elec, label_path, subject_list, path

N_FBIN = 90
WINDOW = 1000
OVERLAP = 0
N_PERMUTATIONS = 1000
SLEEP_LIST = ['S1', 'S2', 'SWS', 'Rem', 'NREM']
DATA_PATH = path('/home/arthur/Documents/riemann/PSD/')
SAVE_PATH = DATA_PATH / 'results'
SUB_LIST = ['s' + str(e) for e in subject_list]
PERM_TEST = False

if __name__ == '__main__':
    for state in SLEEP_LIST:
        print(state)
        for elec in range(n_elec):
            print(elec)
            y = loadmat(label_path / state + '_labels.mat')['y'].ravel()
            y, groups = create_groups(y)

            fbin_not_done = []
            for fbin in range(N_FBIN):
                results_file_path = SAVE_PATH / \
                        'perm_PSD_bin_{}_{}_{}_{}_{:.2f}.mat'.format(
                            fbin, state, elec, WINDOW, OVERLAP)
                if not path(results_file_path).isfile():
                    fbin_not_done.append(fbin)

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
            print('frequency bins :', [f+1 for f in fbin_not_done], sep='\n')

            for fbin in fbin_not_done:
                X = dataset[:, fbin].reshape(-1, 1)
                sl2go = StratifiedLeave2GroupsOut()
                clf = LDA()
                perm_scores = []
                pvalue = 0
                good_scores = cross_val_score(cv=sl2go, estimator=clf,
                                              X=X, y=y,
                                              groups=groups)
                good_score = good_scores.mean()
                if PERM_TEST:
                    for perm in range(N_PERMUTATIONS):
                        clf = LDA()
                        perm_set = permutation(len(y))
                        y_perm = y[perm_set]
                        groups_perm = groups[perm_set]
                        perm_scores.append(cross_val_score(cv=sl2go,
                                                           estimator=clf,
                                                           X=X, y=y_perm,
                                                           groups=groups_perm,
                                                           n_jobs=-1).mean())
                    for score in perm_scores:
                        if good_score <= score:
                            pvalue += 1/N_PERMUTATIONS
                    data = {'score': good_score,
                            'pscore': perm_scores,
                            'pvalue': pvalue}
                    print('{} : {:.2f} significatif a p={:.4f}'.format(
                        fbin, good_score, pvalue))

                    results_file_path = SAVE_PATH / \
                        'perm_PSD_bin_{}_{}_{}_{}_{:.2f}.mat'.format(
                            fbin, state, elec, WINDOW, OVERLAP)
                else:
                    data = {'score': good_scores}
                    results_file_path = SAVE_PATH / \
                        'PSD_bin_{}_{}_{}_{}_{:.2f}.mat'.format(
                            fbin, state, elec, WINDOW, OVERLAP)
                savemat(results_file_path, data)
