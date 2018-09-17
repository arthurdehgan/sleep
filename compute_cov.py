"""Computes Crosspectrum matrices and save them.

Author: Arthur Dehgan"""
import os
import numpy as np
from path import Path as path
from joblib import Parallel, delayed

# from pyriemann.estimationmod import CospCovariances
from pyriemann.estimation import Covariances
from scipy.io import savemat, loadmat

SAVE_PATH = path("SAVEPATH")
DATA_PATH = path("DATA_PATH")
SUBJECT_LIST = []
COND_LIST = []
prefix = "covariance"


def load_samples(data_path, sub, cond):
    file_name = "subject{}_tseries_post.mat".format(sub)
    data_m = loadmat(DATA_PATH / file_name)[cond]
    data = data_m.swapaxes(0, 2)
    data = data.swapaxes(1, 2)
    return data


def combine_subjects(cond):
    """Combines crosspectrum matrices from subjects into one."""
    dat, load_list = [], []
    print(cond)
    for sub in SUBJECT_LIST:
        pattern = prefix + "_s{}_{}.mat".format(sub, cond)
        save_pattern = prefix + "_{}.mat".format(cond)
        file_path = path(SAVE_PATH / pattern)
        try:
            data = loadmat(file_path)["data"]
            dat.append(data)
            load_list.append(str(file_path))
        except IOError:
            print(file_path, "not found")
    savemat(SAVE_PATH / save_pattern.format(cond), {"data": np.asarray(dat)})
    for file in load_list:
        os.remove(file)


def compute_cov(cond):
    """Computes the crosspectrum matrices per subjects."""
    for sub in SUBJECT_LIST:
        pattern = prefix + "_s{}_{}.mat".format(sub, cond)
        file_path = path(SAVE_PATH / pattern)

        if not file_path.isfile():
            # data must be of shape n_trials x n_elec x n_samples
            data = load_samples(DATA_PATH, sub, cond)
            cov = Covariances()
            mat = cov.fit_transform(data)
            savemat(file_path, {"data": mat})


if __name__ == "__main__":
    print("computing cov matrices")
    Parallel(n_jobs=-1)(delayed(compute_cov)(cond) for cond in COND_LIST)
    print("combining subjects data")
    Parallel(n_jobs=-1)(delayed(combine_subjects)(cond) for cond in COND_LIST)
