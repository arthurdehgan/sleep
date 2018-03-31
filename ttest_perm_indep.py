"""Function do perform ttest indep with permutations

Author: Arthur Dehgan"""
import numpy as np
from scipy.stats import ttest_ind


def ttest_perm_ind_maxcor(cond1, cond2, n_perm=0, correction='maxstat'
                          equal_var=False, two_tailed=True):
    """ttest indep with permuattions and maxstat correction

    Parameters:
        cond1, cond2: numpy arrays of shape n_subject x n_eletrodes
                      or n_trials x n_electrodes. arrays of data for
                      the independant conditions.

        n_perm: int, 0, number of permutations to do.
               If n_perm = 0 then exaustive permutations will be done.
               It will take exponential time with data size.

        equal_var: bool, False, see scipy.stats.ttest_ind.

        two_tailed: bool, False, if you want two-tailed ttest.

    Returns:
        tval: list, the calculated t-statistics

        p_val: list if two_tailed = False
               tuple(list: pval_right, list: pval_left) otherwisei
               pvalues after permutation test
    """
    tval = ttest_ind(cond1, cond2, equal_var=equal_var)[0]

    perm_t = perm_test(cond1, cond2, n_perm, equal_var)

    pval = compute_maxstat_pval(tval, perm_t, two_tailed)

    return tval, pval


def perm_test(cond1, cond2, n_perm, equal_var):
    """permuattion ttest.

    Parameters:
        cond1, cond2: numpy arrays of shape n_subject x n_eletrodes
                      or n_trials x n_electrodes. arrays of data for
                      the independant conditions.

        n_perm: int, number of permutations to do, the more the better.

        equal_var: bool, False, see scipy.stats.ttest_ind.

    Returns:
        perm_t: list of permutation t-statistics
    """
    full_mat = np.concatenate((cond1, cond2), axis=0)
    perm_t = []
    if n_perm == 0:
        # for id in combinations(range(len(cond1) + len(cond2)), min(len(cond1), len(cond2))):
            # perm_t.append(ttest_ind(cond1[
    else:
        for _ in range(n_perm):
            np.random.shuffle(full_mat)
            cond1 = full_mat[:len(cond1)]
            cond2 = full_mat[len(cond2):]
            perm_t.append(ttest_ind(cond1, cond2, equal_var=equal_var)[0])
    return perm_t


def compute_maxstat_pval(tval, perm_t, two_tailed):
    """computes pvalues with maxstat correction.

    Parameters:
        tstat: computed t-statistics

        perm_t: list of permutation t-statistics

        two_tailed: bool, False, if you want two-tailed ttest.

    Returns:
        pvalues: list if two_tailed = False
                 tuple(list: pval_right, list: pval_left) otherwise
                 pvalues after permutation test
    """
    pvalues_right = []
    max_t_perm = np.asarray(perm_t).max(axis=1)
    if two_tailed:
        pvalues_left = []
        min_t_perm = np.asarray(perm_t).min(axis=1)

    for tstat in tval:
        p_final_right = 0
        for t_perm in max_t_perm:
            if tstat >= t_perm:
                p_final_right += 1/(len(perm_t)+1)
        pvalues_right.append(float('{:.3f}'.format(p_final_right)))

        if two_tailed:
            p_final_left = 0
            for t_perm in min_t_perm:
                if tstat <= t_perm:
                    p_final_left += 1/(len(perm_t)+1)
            pvalues_left.append(float('{:.3f}'.format(p_final_left)))

    pvalues = (pvalues_left, pvalues_right) if two_tailed\
        else pvalues_right

    return pvalues


if __name__ == '__main__':
    cond1 = np.random.random((10, 10))
    cond2 = np.random.random((10, 10))
    print(cond1)
    print(cond2)
    n_perm = 10
    print(ttest_perm_ind_maxcor(cond1, cond2, n_perm))
