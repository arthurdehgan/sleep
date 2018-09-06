'''Uses a classifier to decode PSD values.

Computes pvalues and saves them in a mat format with the decoding accuracies.

Author: Arthur Dehgan
'''
from time import time
from itertools import product
import numpy as np
from scipy.io import savemat, loadmat
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from utils import StratifiedLeave2GroupsOut, elapsed_time, create_groups,\
                  classification
from params import SAVE_PATH, LABEL_PATH, path, CHANNEL_NAMES,\
                   WINDOW, OVERLAP, STATE_LIST, FREQ_DICT

N_PERMUTATIONS = 1000
SAVE_PATH = SAVE_PATH / 'psd'


def main(state, elec):
    labels = loadmat(LABEL_PATH / state + '_labels.mat')['y'].ravel()
    labels, groups = create_groups(labels)
    final_data = None

    print(state, elec)
    results_file_path = SAVE_PATH / 'results' /\
        'perm_PSDM_{}_{}_{}_{:.2f}_NoGamma.mat'.format(
            state, elec, WINDOW, OVERLAP)

    if not path(results_file_path).isfile():
        # print('\nloading PSD for {} frequencies'.format(key))

        for key in FREQ_DICT:
            if not key.startswith('Gamma'):
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
        save = classification(clf, sl2go, final_data,
                              labels, groups, n_jobs=-1)

        savemat(results_file_path, save)


if __name__ == '__main__':
    T0 = time()

    for state, elec in product(STATE_LIST, CHANNEL_NAMES):
        main(state, elec)

    print('total time lapsed : {}'.format(elapsed_time(T0, time())))
