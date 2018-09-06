"""Computes Crosspectrum matrices and save them.

Author: Arthur Dehgan"""
import os
from time import time
from itertools import product
from joblib import Parallel, delayed
import numpy as np
from scipy.io import savemat, loadmat
from utils import elapsed_time

# from utils import load_samples
from utils import load_full_sleep, load_samples
from params import (
    DATA_PATH,
    SAVE_PATH,
    SUBJECT_LIST,
    FREQ_DICT,
    STATE_LIST,
    SF,
    WINDOW,
    OVERLAP,
)

IMAG = False
FULL_TRIAL = False
if IMAG:
    from pyriemann.estimationmod import CospCovariances
else:
    from pyriemann.estimation import CospCovariances
if IMAG:
    prefix = "im_cosp"
elif FULL_TRIAL:
    prefix = "ft_cosp"
else:
    prefix = "cosp"

SAVE_PATH = SAVE_PATH / "cosp/"


def combine_subjects(state, freq, window, overlap, cycle=None):
    """Combines crosspectrum matrices from subjects into one."""
    dat, load_list = [], []
    print(state, freq)
    for sub in SUBJECT_LIST:
        file_path = SAVE_PATH / prefix + "_s{}_{}_{}_{}_{:.2f}.mat".format(
            sub, state, freq, window, overlap
        )
        save_file_path = SAVE_PATH / prefix + "_{}_{}_{}_{:.2f}.mat".format(
            state, freq, window, overlap
        )
        if cycle is not None:
            file_path = SAVE_PATH / prefix + "_s{}_{}_cycle{}_{}_{}_{:.2f}.mat".format(
                sub, state, cycle, freq, window, overlap
            )
            save_file_path = SAVE_PATH / prefix + "_{}_cycle{}_{}_{}_{:.2f}.mat".format(
                state, cycle, freq, window, overlap
            )
        try:
            data = loadmat(file_path)["data"]
            dat.append(data)
            load_list.append(str(file_path))
        except (IOError, TypeError) as e:
            print(file_path, "not found")
    savemat(save_file_path, {"data": np.asarray(dat)})
    for f in load_list:
        os.remove(f)


def compute_cosp(state, freq, window, overlap, cycle=None):
    """Computes the crosspectrum matrices per subjects."""
    if cycle is not None:
        print(state, freq, cycle)
    else:
        print(state, freq)
    freqs = FREQ_DICT[freq]
    for sub in SUBJECT_LIST:
        if cycle is None:
            file_path = SAVE_PATH / prefix + "_s{}_{}_{}_{}_{:.2f}.mat".format(
                sub, state, freq, window, overlap
            )
        else:
            file_path = SAVE_PATH / prefix + "_s{}_{}_cycle{}_{}_{}_{:.2f}.mat".format(
                sub, state, cycle, freq, window, overlap
            )

        if not file_path.isfile():
            # data must be of shape n_trials x n_elec x n_samples
            if cycle is not None:
                data = load_full_sleep(DATA_PATH, sub, state, cycle)
                if data is None:
                    continue
                data = data.swapaxes(1, 2)
            else:
                data = load_samples(DATA_PATH, sub, state)
            if FULL_TRIAL:
                data = np.concatenate(data, axis=1)
                data = data.reshape(1, data.shape[0], data.shape[1])
            cov = CospCovariances(
                window=window, overlap=overlap, fmin=freqs[0], fmax=freqs[1], fs=SF
            )
            mat = cov.fit_transform(data)
            if len(mat.shape) > 3:
                mat = np.mean(mat, axis=-1)

            savemat(file_path, {"data": mat})


if __name__ == "__main__":
    T_START = time()
    Parallel(n_jobs=-1)(
        delayed(compute_cosp)(state, freq, WINDOW, OVERLAP, cycle)
        for state, freq, cycle in product(STATE_LIST, FREQ_DICT, range(1, 4))
    )
    print("combining subjects data")
    Parallel(n_jobs=-1)(
        delayed(combine_subjects)(state, freq, WINDOW, OVERLAP, cycle)
        for state, freq, cycle in product(STATE_LIST, FREQ_DICT, range(1, 4))
    )
    print("total time lapsed : %s" % elapsed_time(T_START, time()))
