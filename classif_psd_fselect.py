'''Uses a classifier to decode PSD values.

Computes pvalues and saves them in a mat format with the decoding accuracies.

Author: Arthur Dehgan
'''
from time import time
import numpy as np
from scipy.io import savemat, loadmat
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from utils import StratifiedLeave2GroupsOut, elapsed_time
from params import SAVE_PATH, path, CHANNEL_NAMES,\
                   WINDOW, OVERLAP, STATE_LIST, FREQ_DICT

N_PERMUTATIONS = 1000
SAVE_PATH = SAVE_PATH / 'psd'
FREQS = np.array(['delta', 'theta', 'alpha', 'sigma',
                  'beta', 'gamma1', 'gamma2'])


def create_groups(y):
    """Generate groups from labels of shape (subject x labels)."""
    k = 0
    groups = []
    for i in range(len(y)):
        for j in range(y[i].shape[0]):
            groups.append(k)
        k += 1
    groups = np.asarray(groups).ravel()
    y = np.concatenate(y[:]).ravel()
    return y, groups


def classification(state, elec):
    final_data = None

    print(state, elec)
    results_file_path = SAVE_PATH / 'results' /\
        'EFS_NoGamma_{}_{}_{}_{:.2f}.mat'.format(state, elec, WINDOW, OVERLAP)
    if not path(results_file_path).isfile():
        # print('\nloading PSD for {} frequencies'.format(key))
        for key in FREQ_DICT:
            data_file_path = SAVE_PATH /\
                    'PSD_{}_{}_{}_{}_{:.2f}.mat'.format(
                        state, key, elec, WINDOW, OVERLAP)

            if path(data_file_path).isfile():
                data = loadmat(data_file_path)['data'].ravel()
                if final_data is None:
                    final_data = data
                else:
                    for i, submat in enumerate(final_data):
                        final_data[i] = np.concatenate((
                            submat, data[i]), axis=0)
            else:
                print(path(data_file_path).name + ' Not found')
                print('please run "computePSD.py" and\
                      "group_PSD_per_subjects.py" before\
                      running this script')

        lil_labels = [0]*18 + [1]*18
        lil_groups = list(range(36))
        sl2go = StratifiedLeave2GroupsOut()

        best_freqs = []
        best_scores = []
        test_scores = []
        for train_subjects, test_subjects in sl2go.split(final_data, lil_labels, lil_groups):
            X_feature, X_classif = final_data[train_subjects], final_data[test_subjects]
            y_feature, y_classif = lil_labels[train_subjects], lil_labels[test_subjects]

            labels = [np.array([label]*X_feature[i].shape[1])
                      for i, label in enumerate(y_feature)]
            labels, groups = create_groups(labels)
            X_feature = np.concatenate(X_feature[:], axis=1).T

            nested_sl2go = StratifiedLeave2GroupsOut()
            clf = LDA()
            f_select = EFS(estimator=clf,
                           max_features=X_feature.shape[-1],
                           cv=nested_sl2go,
                           n_jobs=-1)

            f_select = f_select.fit(X_feature,
                                    labels,
                                    groups)

            best_idx = f_select.best_idx_
            best_freqs.append(FREQS[list(best_idx)])
            best_scores.append(f_select.best_score_)

            test_clf = LDA()
            test_score = test_clf.score(X_classif, y_classif)
            test_scores.append(test_score)

        score = np.mean(test_scores)
        data = {'score': score,
                'train_scores': best_scores,
                'test_scores': test_scores,
                'freqs': best_freqs}
        print('\nBest combi: {} - {:.2f}'.format(best_freqs, score))

        savemat(results_file_path, data)


if __name__ == '__main__':
    T0 = time()

    for state, elec in product(STATE_LIST, CHANNEL_NAMES):
        main(state, elec)

    print('total time lapsed : {}'.format(elapsed_time(T0, time())))
