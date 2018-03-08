from scipy.io import loadmat, savemat
from params import subject_list as sub_list
import numpy as np

DP = '/home/arthur/Documents/riemann/PSD/'
sl = ['S1', 'S2', 'SWS']
freql = ['Alpha', 'Theta', 'Beta', 'Gamma1', 'Gamma2', 'Delta', 'Sigma']

# for freq in freql:
for sub in sub_list:
    for e in range(19):
        l, final = [], []
        for s in sl:
            fn = DP + 'PSDs_{}_s{}_{}_1000_0.00.mat'.format(s, sub, e)
            temp = loadmat(fn)['data']
#             fn = DP + 'PSD_sleepState_{}_{}_{}_1000_0.00.mat'.format(s, freq, e)
#             temp = loadmat(fn)['data'].ravel().tolist()
#             for i, j in enumerate(temp):
#                 temp[i] = j.ravel().tolist()
            l.append(temp)
#         for i in range(len(l[0])):
#             final.append(l[0][i] + l[1][i] + l[2][i])
        # fn2 = DP + 'PSD_sleepState_NREM_{}_{}_1000_0.00.mat'.format(freq, e)
        final = np.vstack(l)
        fn2 = DP + 'PSDs_NREM_s{}_{}_1000_0.00.mat'.format(sub, e)
        savemat(fn2, {'data': final})
