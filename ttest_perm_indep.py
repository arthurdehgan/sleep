"""Function do perform ttest indep with permutations

Author: Arthur Dehgan"""
import numpy as np
from scipy.stats import ttest_ind
from scipy.special import comb
from itertools import combinations
from joblib import Parallel, delayed
from sys import maxsize


def ttest_perm_unpaired(cond1, cond2, n_perm=0, correction='maxstat',
                        method='indep', alpha=0.05, equal_var=False, two_tailed=False, n_jobs=1):
    """ttest indep with permuattions and maxstat correction

    Parameters:
        cond1, cond2: numpy arrays of shape n_subject x n_eletrodes
                      or n_trials x n_electrodes. arrays of data for
                      the independant conditions.

        n_perm: int, number of permutations to do.
                If n_perm = 0 then exaustive permutations will be done.
                It will take exponential time with data size.

        correction: string, None, the choice of correction to compute
                    pvalues. If None, no correction will be done
                    Options are 'maxstat', 'fdr', 'bonferroni', None

        method : 'indep' | 'negcorr'
                Necessary only for fdr correction.
                Implements Benjamini/Hochberg method if 'indep' or
                Benjamini/Yekutieli if 'negcorr'.

        alpha: float, error rate

        equal_var: bool, see scipy.stats.ttest_ind.

        two_tailed: bool, set to True if you want two-tailed ttest.

        n_jobs: int, Number of cores used to computer permutations in
                parallel (-1 uses all cores and will be faster)

    Returns:
        tval: list, the calculated t-statistics

        pval: pvalues after permutation test and correction if selected
    """
    if correction not in ['maxstat', 'bonferroni', 'fdr', None]:
        raise ValueError(correction, 'is not a valid correction option')

    tval = ttest_ind(cond1, cond2, equal_var=equal_var)[0]

    perm_t = perm_test(cond1, cond2, n_perm, equal_var, n_jobs=n_jobs)

    pval = compute_pvalues(tval, perm_t, two_tailed,
                           correction=correction, method=method)

    if correction in ['bonferroni', 'fdr']:
        pval = pvalues_correction(pval, correction, method)

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

        equal_var: bool, see scipy.stats.ttest_ind.

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


def compute_pvalues(tval, perm_t, two_tailed, correction, method):
    """computes pvalues without any correction.

    Parameters:
        tstat: computed t-statistics

        perm_t: list of permutation t-statistics

        two_tailed: bool, if you want two-tailed ttest.

        correction: string, None, the choice of correction to compute
                    pvalues. If None, no correction will be done
                    Options are 'maxstat', 'fdr', 'bonferroni', None

        method : 'indep' | 'negcorr'
                Necessary only for fdr correction.
                Implements Benjamini/Hochberg method if 'indep' or
                Benjamini/Yekutieli if 'negcorr'.

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
        perm_t = np.array([perm_t for _ in range(len(tval))]).T

    for i, tstat in enumerate(tval):
        p_final = 0
        compare_list = perm_t[:, i]
        for t_perm in compare_list:
            if tstat <= t_perm:
                p_final += 1/scaling
        pvalues.append(p_final)

    pvalues = np.asarray(pvalues, dtype=np.float32)

    return pvalues


def pvalues_correction(pvalues, correction, method):
    if correction == 'bonferroni':
        pvalues *= float(np.array(pvalues).size)

    elif correction == 'fdr':
        n_obs = len(pvalues)
        index_sorted_pvalues = np.argsort(pvalues)
        sorted_pvalues = pvalues[index_sorted_pvalues]
        sorted_index = index_sorted_pvalues.argsort()
        ecdf = (np.arange(n_obs) + 1) / float(n_obs)

        if method == 'negcorr':
            cm = np.sum(1. / (np.arange(n_obs) + 1))
            ecdf /= cm
        elif method == 'indep':
            pass
        else:
            raise ValueError(method, ' is not a valid method option')

        raw_corrected_pvalues = sorted_pvalues / ecdf
        corrected_pvalues = np.minimum.accumulate(
            raw_corrected_pvalues[::-1])[::-1]
        pvalues = corrected_pvalues[sorted_index].reshape(n_obs)

    pvalues[pvalues > 1.0] = 1.0

    return pvalues


if __name__ == '__main__':
    cond1 = np.random.randn(10, 10)
    cond2 = np.random.randn(10, 10)
    tval, pval = ttest_perm_unpaired(cond1, cond2, n_perm=100)
    tval2, pval2 = ttest_perm_unpaired(cond1, cond2, n_perm=100, correction='bonferroni')
    tval3, pval3 = ttest_perm_unpaired(cond1, cond2, n_perm=100, correction='fdr')
    print(pval ,pval2, pval3, sep='\n')
