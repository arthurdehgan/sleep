from scipy.io import loadmat, savemat
from params import CHANNEL_NAMES, FREQ_DICT
import numpy as np

DP = '/home/arthur/Documents/sleep/features/crosspectre/'
# DP = '/home/arthur/Documents/sleep/features/psd/'
# DP = '/home/arthur/Documents/sleep/features/covariance/'
LP = '/home/arthur/Documents/sleep/labels/'
STATE_LIST = ['S1', 'S2', 'SWS']


# labels = []
# for state in STATE_LIST:
    # label_path = LP + '{}_labels.mat'.format(state)
    # temp_label = loadmat(label_path)['y'].ravel()
    # temp_label = [lab.ravel() for lab in temp_label]
    # labels.append(temp_label)
#
# labelsf = []
# for i in range(len(labels[0])):
    # temp = []
    # for k in range(len(labels)):
        # temp.append(labels[k][i])
    # temp = np.concatenate(temp)
    # labelsf.append(temp)
#
# labelsf = np.array(labelsf)
# savemat(LP + 'NREM_labels.mat', {'y': labelsf})


for key in FREQ_DICT:
    # for elec in CHANNEL_NAMES:
    data = []
    for state in STATE_LIST:
        # file_name = DP + 'PSD_{}_{}_{}_1000_0.00.mat'.format(state, key, elec)
        file_name = DP + 'im_cosp_{}_{}_1000_0.00.mat'.format(state, key)
        # file_name = DP + 'cov_{}.mat'.format(state)
        temp_data = loadmat(file_name)['data'].ravel()
        # temp_data = [dat.ravel() for dat in temp_data]
        data.append(temp_data)
    dataf = []
    for i in range(len(data[0])):
        temp = []
        for k in range(len(STATE_LIST)):
            temp.append(data[k][i])
        temp = np.concatenate(temp)
        dataf.append(temp)
    dataf = np.array(dataf)
    # savemat(DP + 'PSD_NREM_{}_{}_1000_0.00.mat'.format(key, elec), {'data': dataf})
    # savemat(DP + 'cov_NREM.mat', {'data': dataf})
    savemat(DP + 'im_cosp_NREM_{}_1000_0.00.mat'.format(key), {'data': dataf})

