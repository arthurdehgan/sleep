"""Computes PSD vectors and save them.

Execute "group_PSD_per_subjects.py" after this script

Author: Arthur Dehgan
"""
from utils import load_samples, elapsed_time, computePSD
from scipy.io import savemat
from time import time
import numpy as np
from path import Path as path
from joblib import Parallel, delayed
from params import DATA_PATH, SAVE_PATH, SUBJECT_LIST, FREQ_DICT, STATE_LIST,\
                   SF, WINDOW, OVERLAP, CHANNEL_NAMES


SAVE_PATH += 'psd'


def computeAndSavePSD(SUBJECT_LIST, state, freq, window, overlap, fmin, fmax,
                      fs, elec=None):
    '''loads data, compute PSD and saves PSD of all subjects in one file'''
    N_ELEC = 19 if elec is None else len(elec)
    print(state, freq, 'bande {}: [{}-{}]Hz'.format(freq, fmin, fmax))
    for elec in range(N_ELEC):  # pour chaque elec
        channel_name = CHANNEL_NAMES[elec]
        file_path = path(SAVE_PATH /
                         'PSD_%s_%s_%i_%i_%.2f.mat' %
                         # 'PSD_EOG_sleepState_%s_%s_%i_%i_%.2f.mat' %
                         (state, freq, channel_name, window, overlap))
        if not file_path.isfile():
            psds = []
            for sub in SUBJECT_LIST:  # pour chaque sujet
                X = load_samples(DATA_PATH, sub, state)
                psd_list = []
                master_psd = []
                for j in range(X.shape[0]):  # pour chaque trial
                    psd = computePSD(X[j, elec], window=window,
                                     overlap=OVERLAP,
                                     fmin=fmin,
                                     fmax=fmax,
                                     fs=fs)
                    psd_list.append(psd)
                master_psd.append(psd_list)
                master_psd = np.asarray(master_psd)
                psds.append(master_psd.ravel())

            savemat(file_path, {'data': psds})


if __name__ == '__main__':
    """Do the thing."""
    t0 = time()

    #Parallel(n_jobs=-1)(delayed(computeAndSavePSD)(SUBJECT_LIST,
    #                                               state,
    #                                               freq=freq,
    #                                               window=WINDOW,
    #                                               overlap=OVERLAP,
    #                                               fmin=FREQ_DICT[freq][0],
    #                                               fmax=FREQ_DICT[freq][1],
    #                                               fs=SF)
    #                    for freq in FREQ_DICT
    #                    for state in STATE_LIST)
    for state in STATE_LIST:
        for freq in FREQ_DICT:
            computeAndSavePSD(SUBJECT_LIST, state, freq, WINDOW, OVERLAP,
                              fmin=FREQ_DICT[freq][0],
                              fmax=FREQ_DICT[freq][1],
                              fs=SF)
    print('total time lapsed : %s' % elapsed_time(t0, time()))
