"""Computes Covariance matrices and save them."""
from time import time
from scipy.io import savemat
from pyriemann.estimation import Covariances
from path import Path as path
from joblib import Parallel, delayed
from utils import import_data, elapsed_time
from params import DATA_PATH, SAVE_PATH, SUBJECT_LIST, LABEL_PATH, STATE_LIST


FULL_TRIAL = False
if FULL_TRIAL:
    SAVE_PATH += 'covariance_full_trial'
else:
    SAVE_PATH += 'covariance'


def main(state):
    """Do the thing."""

    file_path = path(SAVE_PATH / 'cov_{}.mat'.format(state))
    if not file_path.isfile():
        data, _ = import_data(DATA_PATH, state,
                              SUBJECT_LIST, LABEL_PATH, FULL_TRIAL)

        covs = []
        for trial in data:
            cov = Covariances()
            covs.append(cov.fit_transform(trial))

        savemat(file_path, {'data': covs})


if __name__ == '__main__':
    START_TIME = time()
    Parallel(n_jobs=-1)(delayed(main)(state) for state in STATE_LIST)
    print('total time lapsed : %s' % elapsed_time(START_TIME, time()))
