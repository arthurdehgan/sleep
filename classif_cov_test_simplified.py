"""Load covariance matrix, perform classif, perm test, saves results.

Outputs one file per freq x state

Author: Arthur Dehgan"""
from time import time
from scipy.io import savemat, loadmat
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score
from pyriemann.classification import TSclassifier
from utils import StratifiedLeave2GroupsOut, prepare_data, classification
from params import SAVE_PATH, STATE_LIST

prefix = "classif_subsamp_"
name = "cov"
state = "SWS"

SAVE_PATH = SAVE_PATH / name

info_data = pd.read_csv(SAVE_PATH.parent / "info_data.csv")[STATE_LIST]
info_data = info_data[state]
N_TRIALS = info_data.min().min()
N_SUBS = len(info_data) - 1
groups = [i for i in range(N_SUBS) for _ in range(N_TRIALS)]
N_TOTAL = N_TRIALS * N_SUBS
labels = [0 if i < N_TOTAL / 2 else 1 for i in range(N_TOTAL)]

file_name = prefix + name + "n153_{}.mat".format(state)

save_file_path = SAVE_PATH / "results" / file_name

data_file_path = SAVE_PATH / name + "_{}.mat".format(state)

final_save = None

data = loadmat(data_file_path)
data = prepare_data(data, n_trials=N_TRIALS, random_state=0)

sl2go = StratifiedLeave2GroupsOut()
lda = LDA()
clf = TSclassifier(clf=lda)
score = cross_val_score(clf, data, labels, groups, cv=sl2go, n_jobs=-1)
print(score)
# save['acc_bootstrap'] = [save['acc_score']]
# save['auc_bootstrap'] = [save['auc_score']]
# if final_save is None:
#     final_save = save
# else:
#     for key, value in final_save.items():
#         final_save[key] = final_save[key] + save[key]

# savemat(save_file_path, final_save)
