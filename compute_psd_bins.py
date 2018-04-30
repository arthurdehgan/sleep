"""Computes PSD vectors and save them.

Computes PSD for each frequency from meg fif files and saves.

"""
from scipy.io import savemat, loadmat
from scipy.signal import welch
from path import Path as path
from joblib import Parallel, delayed
import numpy as np
from params import STATE_LIST, SUBJECT_LIST, SF, N_ELEC,\
                   WINDOW, OVERLAP, DATA_PATH, SAVE_PATH, CHANNEL_NAMES

SAVE_PATH = SAVE_PATH / 'psd/'
freq_range = (1, 45)
del STATE_LIST[STATE_LIST.index('NREM')]


def computePSD(signal, fs, window, overlap, freq_range):
    f, psd = welch(signal, fs=fs, window='hamming', nperseg=window,
                   noverlap=overlap, nfft=None)
    return psd[(f >= freq_range[0])*(f <= freq_range[1])]


def computeSavePSD(sleep_stages, subject, window, overlap):
    for stage in sleep_stages:
        print(stage)
        file_name = DATA_PATH / '{}_s{}.mat'.format(stage, subject)
        X = loadmat(file_name)[stage][:N_ELEC].swapaxes(1, 2)
        for i, elec in enumerate(CHANNEL_NAMES):
            psd = []
            for trial in X[i]:
                save_name = SAVE_PATH / 'PSDs_{}_s{}_{}_{}_{:.2f}.mat'.format(
                    stage, subject, elec, window, overlap)
                psd.append(computePSD(trial, SF, window, overlap, freq_range))
            savemat(save_name, {'data': np.asarray(psd)})


if __name__ == '__main__':
    """Main function."""

    Parallel(n_jobs=1)(delayed(computeSavePSD)(STATE_LIST,
                                               subject,
                                               window=WINDOW,
                                               overlap=OVERLAP) for subject in SUBJECT_LIST)
        #computeSavePSD(STATE_LIST, subject, window=WINDOW, overlap=OVERLAP)
