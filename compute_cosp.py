"""Computes Crosspectrum matrices and save them.

Author: Arthur Dehgan"""
import os
from time import time
from path import Path as path
from joblib import Parallel, delayed
import numpy as np
# from pyriemann.estimationmod import CospCovariances
from pyriemann.estimation import CospCovariances
from scipy.io import savemat, loadmat
from utils import elapsed_time, load_samples
from params import DATA_PATH, SAVE_PATH, SUBJECT_LIST, \
                   FREQ_DICT, STATE_LIST, SF, WINDOW, OVERLAP

SAVE_PATH = SAVE_PATH / 'crosspectre/'


def combine_subjects(state, freq, window, overlap):
    """Combines crosspectrum matrices from subjects into one."""
    dat, load_list = [], []
    for sub in SUBJECT_LIST:
        # file_path = path(SAVE_PATH / 'im_cosp_s{}_{}_{}_{}_{:.2f}.mat'.format(
        file_path = path(SAVE_PATH / 'cosp_s{}_{}_{}_{}_{:.2f}.mat'.format(
            sub, state, freq, window, overlap))
        try:
            data = loadmat(file_path)['data']
            dat.append(data)
            load_list.append(str(file_path))
        except IOError:
            print(file_path, "not found")
        path_len = len(SAVE_PATH)
    # savemat(file_path[:path_len + 7] + file_path[path_len + 11:],
    savemat(file_path[:path_len + 4] + file_path[path_len + 8:],
            {'data': np.asarray(dat)})
    for f in load_list:
        os.remove(f)


def compute_cosp(state, freq, window, overlap):
    """Computes the crosspectrum matrices per subjects."""
    print(state, freq)
    freqs = FREQ_DICT[freq]
    for sub in SUBJECT_LIST:
        # file_path = path(SAVE_PATH / 'im_cosp_s{}_{}_{}_{}_{:.2f}.mat'.format(
        file_path = path(SAVE_PATH / 'cosp_s{}_{}_{}_{}_{:.2f}.mat'.format(
            sub, state, freq, window, overlap))

        if not file_path.isfile():
            # data must be of shape n_trials x n_elec x n_samples
            data = load_samples(DATA_PATH, sub, state)
            cov = CospCovariances(window=window, overlap=overlap,
                                  fmin=freqs[0], fmax=freqs[1], fs=SF)
            mat = cov.fit_transform(data)
            if len(mat.shape) > 3:
                mat = np.mean(mat, axis=-1)

            savemat(file_path, {'data': mat})


if __name__ == '__main__':
    T_START = time()
    Parallel(n_jobs=-1)(delayed(compute_cosp)(
        state, freq, WINDOW, OVERLAP)
                        for state in STATE_LIST
                        for freq in FREQ_DICT)
    Parallel(n_jobs=-1)(delayed(combine_subjects)(
        state, freq, WINDOW, OVERLAP)
                        for state in STATE_LIST
                        for freq in FREQ_DICT)
    print('total time lapsed : %s' % elapsed_time(T_START, time()))
