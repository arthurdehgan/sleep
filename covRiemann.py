"""Load covariance matrix, perform classif, perm test, saves results."""
from utils import create_groups, classif_choice, StratifiedLeave2GroupsOut, \
                  elapsed_time
from scipy.io import savemat, loadmat
from time import time
import numpy as np
from sklearn.model_selection import cross_val_score
from numpy.random import permutation
from path import Path as path
from params import save_path, sleep_list, label_path


if __name__ == '__main__':
    t0 = time()
    save_path += 'covariance'
    full_trial = False
    permutation_test = True
    if permutation_test:
        ntype = 'perm'
        n_permutations = 1000
    else:
        ntype = 'result'
        n_permutations = 1

    classifier = 'LDA'

    print('\nClassification of dreamers vs non dreamers in Riemanian space')
    print('features : Covariance matrices')
    print('Classifier : ' + classifier)
    print('Nb de permutations : ' + str(n_permutations))

    for sleep_state in sleep_list:

        print("\nProcessing state %s" % sleep_state)
        t1 = time()
        if full_trial:
            y = np.concatenate((np.ones(18,), np.zeros(18,)))
            groups = range(36)
        else:
            y = loadmat(label_path / sleep_state + '_labels.mat')['y'].ravel()
            y, groups = create_groups(y)

        result_file_path = path(save_path / 'results' / classifier /
                                '%s_cov_%s.mat' % (ntype, sleep_state))
        if not result_file_path.isfile():
            print('\nloading covariance matrices')
            data_file_path = path(save_path /
                                  'covariance_matrices_%s.mat' % sleep_state)
            if data_file_path.isfile():
                X = loadmat(data_file_path)['data'].ravel()
                X = np.concatenate((X[range(len(X))]))
            else:
                print(data_file_path.name + ' Not found')
                print('please run "computeCov.py" before running this script')

            t3 = time()
            print('Number of samples :', len(y))
            print('Classification...')
            sl2go = StratifiedLeave2GroupsOut()
            accuracies = []
            clf_choice, clf_params = classif_choice(classifier)

            for perm in range(n_permutations):
                clf = clf_choice
                accuracies.append(cross_val_score(clf,
                                                  X=X, y=y,
                                                  cv=sl2go,
                                                  groups=groups,
                                                  n_jobs=-1).mean())

                y = permutation(y)

            pvalue = 0
            if n_permutations > 1:
                for score in accuracies[1:]:
                    if score > accuracies[0]:
                        pvalue += 1/(n_permutations)

            print('%0.2f significatif a p=%0.4f\n' % (accuracies[0], pvalue))
            data = {'data': accuracies, 'pvalue': pvalue}

            savemat(result_file_path, data)

    print('total time lapsed : %s' % elapsed_time(t0, time()))
