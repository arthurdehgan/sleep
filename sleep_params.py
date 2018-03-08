"""Load parameters for all scripts."""
from path import Path as path

DATA_PATH = path("/media/arthur/Windows/Documents and Settings/karim/Documents/MATLAB/c1c2 new/data_par_satde_v6")
SAVE_PATH = path('/home/arthur/Documents/riemann/')
LABEL_PATH = SAVE_PATH / 'labels/'

SUBJECT_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                15, 16, 17, 18, 19, 20, 23, 24, 25, 26, 27, 28,
                29, 30, 31, 32, 33, 34, 35, 36, 37, 38]
STATE_LIST = ['S1', 'S2', 'SWS', 'Rem']
CHANNEL_NAMES = ['Fz', 'Cz', 'Pz', 'C3', 'C4', 'T3', 'T4', 'Fp1', 'Fp2', 'O1',
                 'O2', 'F3', 'F4', 'P3', 'P4', 'FC1', 'FC2', 'CP1', 'CP2']
FREQ_DICT = {'Delta': (2, 4),
             'Theta': (4, 8),
             'Alpha': (8, 13),
             'Sigma': (11, 16),
             'Beta': (17,35),
             'Gamma1': (30, 60),
             'Gamma2': (60, 90)}
SF = 1000  # sampling frequency
N_ELEC = 19  # number electrodes

# Pour calculs PSD
WINDOW = 1000  # fenetre
OVERLAP = 0  # overlap (entre 0 et 1)
FBIN_LIST = list(range(90))
