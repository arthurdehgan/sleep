"""Function do perform ttest indep with permutations

Author: Arthur Dehgan"""
import numpy as np
from scipy.stats import ttest_ind
from scipy.special import comb
from itertools import combinations
from joblib import Parallel, delayed
from sys import maxsize


def ttest_perm_unpaired(cond1, cond2, n_perm=0, correction='maxstat',
                        equal_var=False, two_tailed=False, n_jobs=1):
    """ttest indep with permuattions and maxstat correction

    Parameters:
        cond1, cond2: numpy arrays of shape n_subject x n_eletrodes
                      or n_trials x n_electrodes. arrays of data for
                      the independant conditions.

        n_perm: int, 0, number of permutations to do.
                If n_perm = 0 then exaustive permutations will be done.
                It will take exponential time with data size.

        correction: string, None, the choice of correction to compute
                    pvalues. If None, no correction will be done
                    Options are 'maxstat', 'FDR', None

        equal_var: bool, False, see scipy.stats.ttest_ind.

        two_tailed: bool, False, set to True if you want two-tailed ttest.

        n_jobs: int, 1, Number of cores used to computer permutations in
                parallel (-1 uses all cores and will be faster)

    Returns:
        tval: list, the calculated t-statistics

        pval: pvalues after permutation test and correction if selected
    """
    tval = ttest_ind(cond1, cond2, equal_var=equal_var)[0]

    perm_t = perm_test(cond1, cond2, n_perm, equal_var, n_jobs=n_jobs)

    pval = compute_pvalues(tval, perm_t, two_tailed)

    return tval, pval


def my_combinations(iterable, r, limit=None):
    i = 0
    for e in combinations(iterable, r):
        yield e
        i += 1
        if limit is not None and i == limit:
            break


def do_ttest_perm(data, index, equal_var):
    index = list(index)
    index_comp = list(set(range(len(data))) - set(index))
    perm_mat = np.vstack((data[index], data[index_comp]))
    cond1, cond2 = perm_mat[:len(index)], perm_mat[len(index):]
    return ttest_ind(cond1, cond2, equal_var=equal_var)[0]


def perm_test(cond1, cond2, n_perm, equal_var, n_jobs):
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
    n_samples = len(full_mat)
    perm_t = []
    n_comb = comb(n_samples, len(cond1))
    if np.isinf(n_comb):
        n_comb = maxsize
    else:
        n_comb = int(n_comb)

    if n_perm == 0 or n_perm >= n_comb - 1:
        # print("All permutations will be done. n_perm={}".format(n_comb - 1))
        n_perm = n_comb
    if n_perm > 9999:
        print('Warning: permutation number is very high : {}'.format(n_perm))
        print('it might take a while to compute ttest on all permutations')
    # else:
        # print("{} permutations will be done".format(n_perm))

    perms_index = my_combinations(range(n_samples), len(cond1), n_perm)
    perm_t = Parallel(n_jobs=n_jobs)(delayed(do_ttest_perm)
                                     (full_mat, index, equal_var)
                                     for index in perms_index)

    return perm_t[1:]  # the first perm is not a permutation


def compute_pvalues(tval, perm_t, two_tailed, correction=None):
    """computes pvalues without any correction.

    Parameters:
        tstat: computed t-statistics

        perm_t: list of permutation t-statistics

        two_tailed: bool, False, if you want two-tailed ttest.

    Returns:
        pvalues: list if two_tailed = False
                 pvalues after permutation test
    """
    scaling = len(perm_t)
    perm_t = np.array(perm_t)
    pvalues = []
    if two_tailed:
        perm_t = abs(perm_t)

    if correction == 'maxstat':
        perm_t = np.asarray(perm_t).max(axis=1)
    elif correction is None:
        pass
    else:
        raise('This correction option has not been implemented yes')

    for i, tstat in enumerate(tval):
        p_final = 0
        compare_list = perm_t if correction == 'maxstat' else perm_t[:, i]
        for t_perm in compare_list:
            if tstat <= t_perm:
                p_final += 1/scaling
        pvalues.append(p_final)

    return pvalues
