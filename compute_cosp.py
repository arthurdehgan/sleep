"""Computes Crosspectrum matrices and save them.

Author: Arthur Dehgan"""
import os
from time import time
from itertools import product
import numpy as np
from scipy.io import savemat, loadmat
from joblib import Parallel, delayed
from utils import elapsed_time
from path import Path as path
from pyriemann import CospCovariances


DATA_PATH = path("")
SAVE_PATH = path("")
SUBJECT_LIST = []
FREQ_DICT = {"theta": (4, 8), "alpha": (8, 12), "beta": (12, 30)}
COND_LIST = []
SAMPLING_FREQ = 1000
WINDOW = 500
OVERLAP = 0
PREFIX = "cosp"
SAVE_PATH = SAVE_PATH / "cosp/"


def load_samples(data):
    # TODO
    return data


def combine_subjects(cond, freq, window, overlap):
    """Combines crosspectrum matrices from subjects into one."""
    dat, load_list = [], []
    print(cond, freq)
    for sub in SUBJECT_LIST:
        file_path = SAVE_PATH / PREFIX + "_s{}_{}_{}_{}_{:.2f}.mat".format(
            sub, cond, freq, window, overlap
        )
        save_file_path = SAVE_PATH / PREFIX + "_{}_{}_{}_{:.2f}.mat".format(
            cond, freq, window, overlap
        )
        try:
            data = loadmat(file_path)["data"]
            dat.append(data)
            load_list.append(str(file_path))
        except (IOError, TypeError) as e:
            print(file_path, "not found")
            print(e)
    savemat(save_file_path, {"data": np.asarray(dat)})
    for file in load_list:
        os.remove(file)


def compute_cosp(cond, freq, window, overlap):
    """Computes the crosspectrum matrices per subjects."""
    print(cond, freq)
    freqs = FREQ_DICT[freq]
    for sub in SUBJECT_LIST:
        file_path = SAVE_PATH / PREFIX + "_s{}_{}_{}_{}_{:.2f}.mat".format(
            sub, cond, freq, window, overlap
        )

        if not file_path.isfile():
            # data must be of shape n_trials x n_elec x n_samples
            data = load_samples(DATA_PATH, sub, cond)
            cov = CospCovariances(
                window=window,
                overlap=overlap,
                fmin=freqs[0],
                fmax=freqs[1],
                fs=SAMPLING_FREQ,
            )
            mat = cov.fit_transform(data)
            # if len(mat.shape) > 3:
            #     mat = np.mean(mat, axis=-1)

            savemat(file_path, {"data": mat})


if __name__ == "__main__":
    T_START = time()
    Parallel(n_jobs=-1)(
        delayed(compute_cosp)(cond, freq, WINDOW, OVERLAP)
        for cond, freq in product(COND_LIST, FREQ_DICT)
    )
    print("combining subjects data")
    Parallel(n_jobs=-1)(
        delayed(combine_subjects)(cond, freq, WINDOW, OVERLAP)
        for cond, freq in product(COND_LIST, FREQ_DICT)
    )
    print("total time lapsed : %s" % elapsed_time(T_START, time()))
