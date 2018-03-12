"""Computes Crosspectrum matrices and save them.

Author: Arthur Dehgan"""
import os
from time import time
from path import Path as path
from joblib import Parallel, delayed
import numpy as np
# from pyriemann.estimationmod import CospCovariances
from pyriemann.estimation import Covariances
from scipy.io import savemat, loadmat
from utils import elapsed_time, load_samples
from params import DATA_PATH, SAVE_PATH, SUBJECT_LIST, STATE_LIST

SAVE_PATH = SAVE_PATH / 'covariance/'


def combine_subjects(state):
    """Combines crosspectrum matrices from subjects into one."""
    dat, load_list = [], []
    print(state)
    for sub in SUBJECT_LIST:
        file_path = path(SAVE_PATH / 'cov_s{}_{}.mat'.format(
            sub, state))
        try:
            data = loadmat(file_path)['data']
            dat.append(data)
            load_list.append(str(file_path))
        except IOError:
            print(file_path, "not found")
        path_len = len(SAVE_PATH)
    savemat(file_path[:path_len + 4] + file_path[path_len + 8:],
            {'data': np.asarray(dat)})
    for file in load_list:
        os.remove(file)


def compute_cov(state):
    """Computes the crosspectrum matrices per subjects."""
    for sub in SUBJECT_LIST:
        file_path = path(SAVE_PATH / 'cov_s{}_{}.mat'.format(
            sub, state))

        if not file_path.isfile():
            # data must be of shape n_trials x n_elec x n_samples
            data = load_samples(DATA_PATH, sub, state)
            cov = Covariances()
            mat = cov.fit_transform(data)
            savemat(file_path, {'data': mat})


if __name__ == '__main__':
    T_START = time()
    Parallel(n_jobs=-1)(delayed(compute_cov)(
        state)
                        for state in STATE_LIST)
    print('combining subjects data')
    Parallel(n_jobs=-1)(delayed(combine_subjects)(
        state)
                        for state in STATE_LIST)
    print('total time lapsed : %s' % elapsed_time(T_START, time()))
