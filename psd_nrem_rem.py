from utils import StratifiedLeave2GroupsOut, elapsed_time, create_groups
from scipy.io import savemat, loadmat
from time import time
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score
from numpy.random import permutation
from params import data_path, save_path, n_elec, label_path, freq_dict, path


def load_data(data_path, sleep_state, key, elec, window, overlap):
    data_file_path = data_path / 'PSD_sleepState_%s_%s_%i_%i_%.2f.mat' % (sleep_state, key, elec, window, overlap)
    # data_file_path = data_path / 'PSD_EOG_sleepState_%s_%s_%i_%i_%.2f.mat' % (sleep_state, key, elec, window, overlap) # EOG
    if path(data_file_path).isfile():
        X = loadmat(data_file_path)['data'].ravel()
        data = X[0].ravel()
        for submat in X[1:]:
            data = np.concatenate((submat.ravel(), data))
            X = data.reshape(len(data), 1)
        del data
        return X
    else:
        print(path(data_file_path).name + ' Not found')
        print('please run "computePSD.py" and "group_PSD_per_subjects.py" before running this script')
        return 0



if __name__ == '__main__':
    t0 = time()

    window = 1000  # windows for computation of cross-spectrum matrices (in number of samples)
    overlap = 0  # overlap for computation of cross-spectrum matrices (0 = no overlap)
    n_permutations = 1000
    # n_elec=2 # EOG
    sleep_list = ['AWA', 'NREM', 'Rem']
    save_path = save_path / 'PSD'
    classifier = 'LDA'

    print('\nClassification of dreamers vs non dreamers')
    print('features : PSD')
    print('parameters : window = %i overlap = %0.2f' % (window, overlap))
    print('Classifier : ' + classifier)

    for sleep_state in sleep_list:
        print("\nProcessing state %s" % sleep_state)
        for elec in range(n_elec):
            print('electrode : %i/%i' % (elec, n_elec))
            t1 = time()
            y = loadmat(label_path / sleep_state + '_labels.mat')['y'].ravel()
            y, groups = create_groups(y)

            for key in freq_dict:

                results_file_path = save_path / 'results' / 'perm_PSD_%s_%s_%i_%i_%0.2f.mat' % (sleep_state, key, elec, window, overlap)
                # results_file_path = save_path / 'results' / 'perm_PSD_EOG_%s_%s_%i_%i_%0.2f.mat' % (sleep_state, key, elec, window, overlap) # EOG
                if not path(results_file_path).isfile():
                    X = None
                    # print('\nloading PSD for %s frequencies' % key)
                    if sleep_state == 'NREM':
                        for n in ['S1', 'S2', 'SWS']:
                            X = load_data(save_path, n, key, elec, window, overlap) if X is None else X.vstack(load_data(n, key, elec, window, overlap))
                    else:
                        X = load_data(save_path, sleep_state, key, elec, window, overlap)
                    t3 = time()
                    # print('Classification...')
                    sl2go = StratifiedLeave2GroupsOut()
                    clf = LDA()
                    scores = []
                    pvalue = 0
                    good_score = cross_val_score(cv=sl2go, estimator=clf,
                                                 X=X, y=y, groups=groups, n_jobs=-1).mean()
                    for perm in range(n_permutations):
                        clf = LDA()
                        perm_set = permutation(len(y))
                        y_perm = y[perm_set]
                        groups_perm = groups[perm_set]
                        scores.append(cross_val_score(cv=sl2go,
                                                      estimator=clf,
                                                      X=X, y=y_perm,
                                                      groups=groups_perm,
                                                      n_jobs=-1).mean())
                    for score in scores:
                        if good_score <= score:
                            pvalue += 1/n_permutations
                    # print('Done in %s' % elapsed_time(t3, time()))
                    data = {'score': good_score, 'pscore': scores, 'pvalue': pvalue}
                    print('%s : %0.2f significatif a p=%0.4f' % (key, score, pvalue))

                    savemat(results_file_path, data)

    print('total time lapsed : %s' % elapsed_time(t0, time()))
