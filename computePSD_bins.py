"""Computes PSD vectors and save them.

Computes PSD for each frequency from meg fif files and saves.

"""
from scipy.io import savemat, loadmat
from scipy.signal import welch
from path import Path as path
from joblib import Parallel, delayed
import numpy as np
from params import sleep_list, subject_list, sf, n_elec, window, overlap, data_path

save_path = path('/home/arthur/Documents/riemann/PSD')
fs = sf
sleep_stages = ['AWA']
freq_range = (1, 90)


def computePSD(signal, fs, window, overlap, freq_range):
    f, psd = welch(signal, fs=fs, window='hamming', nperseg=window,
                   noverlap=overlap, nfft=None)
    return psd[(f >= freq_range[0])*(f <= freq_range[1])]


def computeSavePSD(sleep_stages, subject, window, overlap):
    for stage in sleep_stages:
        file_name = data_path / '{}_s{}.mat'.format(stage, subject)
        X = loadmat(file_name)[stage][:n_elec].swapaxes(1, 2)
        for elec in range(n_elec):
            psd = []
            for trial in X[elec]:
                save_name = save_path / 'PSDs_{}_s{}_{}_{}_{:.2f}.mat'.format(stage, subject, elec, window, overlap)
                psd.append(computePSD(trial, fs, window, overlap, freq_range))
            savemat(save_name, {'data':np.asarray(psd)})


if __name__ == '__main__':
    """Main function."""

    for subject in subject_list:
        # Parallel(n_jobs=1)(delayed(computeSavePSD)(sleep_stages,
        #                                            subject,
        #                                            window=window,
        #                                            overlap=overlap))
        computeSavePSD(sleep_stages, subject, window=window, overlap=overlap)
