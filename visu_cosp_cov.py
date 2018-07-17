import matplotlib.pyplot as plt
from scipy.io import loadmat
from matplotlib import ticker
from params import STATE_LIST, FREQ_DICT, SAVE_PATH, CHANNEL_NAMES

FIG_PATH = SAVE_PATH.parent / 'figures'
# SAVE_PATH = SAVE_PATH / 'cosp'
SAVE_PATH = SAVE_PATH / 'cov'


def do_matrix(mat, file_name):
    fig, ax = plt.subplots()
    mat = ax.matshow(mat)
    ax.set_xticklabels([0] + CHANNEL_NAMES, rotation=90)
    ax.set_yticklabels([0] + CHANNEL_NAMES)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    file_name = str(file_name)
    fig.colorbar(mat)
    plt.savefig(file_name)
    plt.close()


def prepare_recallers(data):
    HR = data[:18]
    LR = data[18:]
    for i, submat in enumerate(HR):
        HR[i] = submat.mean(axis=0)
    for i, submat in enumerate(LR):
        LR[i] = submat.mean(axis=0)
    HR = HR.mean()
    HR /= HR.max()
    LR = LR.mean()
    LR /= LR.max()
    return HR, LR


if __name__ == '__main__':
    for state in STATE_LIST:
        print(state)
        file_name = SAVE_PATH / 'cov_{}.mat'.format(state)
        data = loadmat(file_name)['data'].ravel()
        HR_cov, LR_cov = prepare_recallers(data)
        # do_matrix(HR, FIG_PATH / 'HR_cov_{}'.format(state))
        # do_matrix(LR, FIG_PATH / 'LR_cov_{}'.format(state))
        for key in FREQ_DICT:
            print(key)
            file_name = SAVE_PATH / 'cosp_{}_{}_1000_0.00.mat'.format(
                state, key)
            data = loadmat(file_name)['data'].ravel()
            HR_cosp, LR_cosp = prepare_recallers(data)
            # do_matrix(HR, FIG_PATH / 'HR' + file_name.name[4:-4] + '.png')
            # do_matrix(LR, FIG_PATH / 'LR' + file_name.name[4:-4] + '.png')
