"""Computes Covariance matrices and save them."""
from utils import import_data, elapsed_time
from scipy.io import savemat
from time import time
from pyriemann.estimation import Covariances
from path import Path as path
from params import data_path, save_path, subject_list, label_path, sleep_list


full_trial = False
if full_trial:
    save_path += 'covariance_full_trial'
else:
    save_path += 'covariance'


def Main():
    """Do the thing."""
    t0 = time()

    print('\nComputing covariance matrices...')
    X = None

    for sleep_state in sleep_list:

        print("\nProcessing state %s" % sleep_state)

        file_path = path(save_path / 'covariance_matrices_%s.mat' %
                         sleep_state)
        if not file_path.isfile():
            if X is None:
                t1 = time()
                X, y = import_data(data_path, sleep_state,
                                   subject_list, label_path, full_trial)
                del y

                print('Done. %i trials loaded in %s' %
                      (len(X), elapsed_time(t1, time())))

            print('\nComputation of the covariance matrices')
            t2 = time()
            covs = []
            for i in range(len(X)):
                cov = Covariances()
                covs.append(cov.fit_transform(X[i]))

            del cov
            print('Done in %s' % elapsed_time(t2, time()))
            savemat(file_path, {'data': covs})
            del covs
        X = None

    print('total time lapsed : %s' % elapsed_time(t0, time()))


if __name__ == '__main__':
    Main()
