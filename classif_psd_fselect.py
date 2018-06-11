'''Uses a classifier to decode PSD values.

Computes pvalues and saves them in a mat format with the decoding accuracies.

Author: Arthur Dehgan
'''
from time import time
import numpy as np
from scipy.io import savemat, loadmat
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from mlxtend import ExhaustiveFeatureSelector as EFS
from utils import StratifiedLeave2GroupsOut, elapsed_time, create_groups
from params import SAVE_PATH, LABEL_PATH, path, CHANNEL_NAMES,\
                   WINDOW, OVERLAP, STATE_LIST, FREQ_DICT

N_PERMUTATIONS = 1000
SAVE_PATH = SAVE_PATH / 'psd'
FREQS = np.array(['delta', 'theta', 'alpha', 'sigma',
                  'beta', 'gamma1', 'gamma2'])


def classification(state, elec):
    labels = loadmat(LABEL_PATH / state + '_labels.mat')['y'].ravel()
    labels, groups = create_groups(labels)
    final_data = None

    print(state, elec)
    results_file_path = SAVE_PATH / 'results' /\
        'EFS_{}_{}_{}_{:.2f}.mat'.format(state, elec, WINDOW, OVERLAP)
    if not path(results_file_path).isfile():
        # print('\nloading PSD for {} frequencies'.format(key))

        for key in FREQ_DICT:
            data_file_path = SAVE_PATH /\
                    'PSD_{}_{}_{}_{}_{:.2f}.mat'.format(
                        state, key, elec, WINDOW, OVERLAP)

            if path(data_file_path).isfile():
                temp = loadmat(data_file_path)['data'].ravel()
                data = temp[0].ravel()
                for submat in temp[1:]:
                    data = np.concatenate((submat.ravel(), data))
                data = data.reshape(len(data), 1)
                final_data = data if final_data is None\
                    else np.hstack((final_data, data))
                del temp
            else:
                print(path(data_file_path).name + ' Not found')
                print('please run "computePSD.py" and\
                      "group_PSD_per_subjects.py" before\
                      running this script')

        # print('classification...')
        sl2go = StratifiedLeave2GroupsOut()
        clf = LDA()
        f_select = EFS(estimator=clf,
                       max_features=7,
                       cv=sl2go,
                       n_jobs=-1)

        data = {'score': f_select.best_score_,
                'index': f_select.best_idx_,
                'data': f_select.subsets_}
        print('{} : {:.2f}'.format(FREQS[f_select.idx_], f_select.best_score_))

        savemat(results_file_path, data)
        final_data = None


if __name__ == '__main__':
    T0 = time()

    # Parallel(n_jobs=-1)(delayed(classification)(state,
    #                                             elec)
    #                     for state in STATE_LIST for elec in CHANNEL_NAMES)
    for state in STATE_LIST:
        for elec in CHANNEL_NAMES:
            classification(state, elec)

    print('total time lapsed : {}'.format(elapsed_time(T0, time())))
