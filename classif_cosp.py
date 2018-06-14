"""Load crosspectrum matrix, perform classif, perm test, saves results.

Outputs one file per freq x state

Author: Arthur Dehgan"""
from time import time
from path import Path as path
from joblib import Parallel, delayed
from scipy.io import savemat, loadmat
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from pyriemann.classification import TSclassifier
from utils import create_groups, StratifiedLeave2GroupsOut, elapsed_time
from params import SAVE_PATH, FREQ_DICT, STATE_LIST, WINDOW,\
                   OVERLAP, LABEL_PATH
# import pdb

# prefix = 'perm_'
prefix = 'classif_'
name = 'cosp'
# name = 'ft_cosp'
# name = 'moy_cosp'
# name = 'im_cosp'
# name = 'wpli'
# name = 'coh'
# name = 'imcoh'
# name = 'ft_wpli'
# name = 'ft_coh'
# name = 'ft_imcoh'
pattern = name + '_{}_{}_{}_{:.2f}.mat'
SAVE_PATH = SAVE_PATH / name
FULL_TRIAL = name.startswith('ft') or name.startswith('moy')


def cross_val(train_index, test_index, clf, X, y):
    clf_copy = clf
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf_copy.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return y_pred, y_test


def cross_val_scores(clf, cv, X, y, groups=None, n_jobs=1):
    results = (Parallel(n_jobs=n_jobs)(delayed(cross_val)(
        train_index,
        test_index,
        clf,
        X,
        y)
        for train_index, test_index in cv.split(X=X, y=y,
                                                groups=groups)))
    accuracy, auc_list = [], []
    for test in results:
        y_pred = test[0]
        y_test = test[1]
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        accuracy.append(acc)
        auc_list.append(auc)
    return accuracy, auc_list


def main(state, key):
    """Where the magic happens"""
    print(state, key)
    if FULL_TRIAL:
        label = np.concatenate((np.ones(18,), np.zeros(18,)))
        groups = range(36)
    else:
        label = loadmat(LABEL_PATH / state + '_labels.mat')['y'].ravel()
        label, groups = create_groups(label)

    file_path = SAVE_PATH / 'results' /\
        prefix + name + '_{}_{}_{}_{:.2f}.mat'.format(
            state, key, WINDOW, OVERLAP)

    if not file_path.isfile():
        data_file_path = path(SAVE_PATH / pattern.format(
            state, key, WINDOW, OVERLAP))
        if data_file_path.isfile():
            data = loadmat(data_file_path)['data']
            # data = data.astype(np.float32)
            if not FULL_TRIAL:
                data = data.ravel()
                data = np.concatenate((data[range(len(data))]))
                if len(data.shape) > 3:
                    data = data.mean(axis=-1)

            # for trial in range(len(data)):
            #     for i in range(19):
            #         for j in range(19):
            #             dat = data[trial,i,j]
            #             if np.isfinite(dat):
            #                 data[trial,i,j] = 0
            #             else:
            #                 data[trial,i,j] = abs(dat)
            # print(data[trial])

            cross_val = StratifiedLeave2GroupsOut()
            lda = LDA()
            clf = TSclassifier(clf=lda)
            accuracy, auc_list = cross_val_scores(clf, cross_val,
                                                  data, label,
                                                  groups, n_jobs=1)

            savemat(file_path, {'data': accuracy, 'auc': auc_list})

            accuracy = np.asarray(accuracy)
            print('accuracy for %s frequencies : %0.2f (+/- %0.2f)' %
                  (key, np.mean(accuracy), np.std(accuracy)))

        else:
            print(data_file_path.name + ' Not found')


if __name__ == '__main__':
    TIMELAPSE_START = time()
    for key in FREQ_DICT:
        for state in STATE_LIST:
            main(state, key)
    print('total time lapsed : %s' % elapsed_time(TIMELAPSE_START, time()))
