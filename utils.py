"""Functions used to compute and analyse EEG/MEG data with pyriemann."""
import time
import functools
from itertools import combinations, permutations
from abc import ABCMeta, abstractmethod
from six import with_metaclass
from sklearn.base import clone
from sklearn.utils import indexable
from sklearn.utils.validation import check_array, _num_samples
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.io import loadmat, savemat
from scipy.signal import welch
import h5py
import numpy as np
from numpy.random import permutation
from path import Path as path
from joblib import Parallel, delayed


def timer(func):
    """Decorator to compute time spend for the wrapped function"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        val = func(*args, **kwargs)
        time_diff = elapsed_time(start_time, time.perf_counter())
        print('"{}" executed in {}'.format(func.__name__, time_diff))
        return val

    return wrapper


def _cross_val(train_index, test_index, estimator, X, y):
    """Computes predictions for a subset of data."""
    clf = clone(estimator)
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return y_pred, y_test


def cross_val_scores(estimator, cv, X, y, groups=None, n_jobs=1):
    """Computes all crossval on the chosen estimator, cross-val and dataset.
    To use instead of sklearn cross_val_score if you want both roc_auc and
    acc in one go."""
    clf = clone(estimator)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_cross_val)(train_index, test_index, clf, X, y)
        for train_index, test_index in cv.split(X=X, y=y, groups=groups)
    )

    accuracy, auc_list = [], []
    for test in results:
        y_pred = test[0]
        y_test = test[1]
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        accuracy.append(acc)
        auc_list.append(auc)
    return accuracy, auc_list


def _permutations(iterable, size, limit=None):
    """Combinations Generator"""
    i = 0
    for elem in permutations(iterable, size):
        yield elem
        i += 1
        if limit is not None and i == limit:
            break


def permutation_test(estimator, cv, X, y, groups=None, n_perm=0, n_jobs=1):
    """Will do compute permutations aucs and accs."""
    acc_pscores, auc_pscores = [], []
    for _ in range(n_perm):
        perm_index = permutation(len(y))
        clf = clone(estimator)
        y_perm = y[perm_index]
        perm_acc, perm_auc = cross_val_scores(clf, cv, X, y_perm, groups, n_jobs)
        acc_pscores.append(np.mean(perm_acc))
        auc_pscores.append(np.mean(perm_auc))

    return acc_pscores, auc_pscores


def classification(estimator, cv, X, y, groups=None, perm=None, n_jobs=1):
    """Do a classification.

    Parameters:
        estimator: a classifier object from sklearn

        cv: a cross-validation object from sklearn

        X: The Data, array of size n_samples x n_features

        y: the labels, array of size n_samples

        groups: optional, groups for groups based cross-validations

        perm: optional, None means no permutations will be computed
            otherwise set her the number of permutations

        n_jobs: optional, default: 1, number of threads to use during
            for the cross-validations. higher means faster. setting to -1 will use
            all available threads - Warning: may sow down computer.

    Returns:
        save: a dictionnary countaining:
            acc_score: the mean score across all cross-validations using the
            accuracy scoring method
            auc_score: the mean score across all cross-validations using the
            roc_auc scoring method
            acc: the list of all cross-validations accuracy scores
            auc: the list of all cross-validations roc_auc scores

        if permutation is not None it also countains:
            auc_pvalue: the pvalue using roc_auc as a scoring method
            acc_pvalue: the pvalue using accuracy as a scoring method
            auc_pscores: a list of all permutation auc scores
            acc_pscores: a list of all permutation accuracy scores

    """
    y = np.asarray(y)
    X = np.asarray(X)
    if len(X) != len(y):
        raise ValueError(
            "Dimension mismatch for X and y : {}, {}".format(len(X), len(y))
        )
    if groups is not None:
        try:
            if len(y) != len(groups):
                raise ValueError("dimension mismatch for groups and y")
        except TypeError:
            print(
                "Error in classification: y or",
                "groups is not a list or similar structure",
            )
            exit()
    clf = clone(estimator)
    accuracies, aucs = cross_val_scores(clf, cv, X, y, groups, n_jobs)
    acc_score = np.mean(accuracies)
    auc_score = np.mean(aucs)
    save = {
        "acc_score": acc_score,
        "auc_score": auc_score,
        "acc": accuracies,
        "auc": aucs,
    }
    if perm is not None:
        acc_pscores, auc_pscores = permutation_test(clf, cv, X, y, groups, perm, n_jobs)
        acc_pvalue = compute_pval(acc_score, acc_pscores)
        auc_pvalue = compute_pval(auc_score, auc_pscores)

        save.update(
            {
                "auc_pvalue": auc_pvalue,
                "acc_pvalue": acc_pvalue,
                "auc_pscores": auc_pscores,
                "acc_pscores": acc_pscores,
            }
        )

    return save


def compute_pval(score, perm_scores):
    """computes pvalue of an item in a distribution)"""
    n_perm = len(perm_scores) + 1
    pvalue = 0
    for psc in perm_scores:
        if score <= psc:
            pvalue += 1 / n_perm
    return pvalue


def gen_dif_index(groups, test_group):
    """Generate an index to identify which group is in test_group."""
    a = list(set(test_group))
    index = []
    for e in a:
        index.append(np.where(groups == e)[0][0])
    return index


def is_strat(y, groups=None, test_group=None):
    """Tell if a label vector is stratified."""
    if groups is not None:
        indx = gen_dif_index(groups, test_group)
        check = len(indx)
    else:
        check = len(y)
    labels = []
    for i in indx:
        labels.append(y[i])
    labels = np.asarray(labels)
    if len(np.where(labels == 1)[0]) == check / 2:
        return True
    else:
        return False


def computePSD(signal, window, overlap, fmin, fmax, fs):
    """Compute PSD."""
    f, psd = welch(
        signal, fs=fs, window="hamming", nperseg=window, noverlap=overlap, nfft=None
    )
    psd = np.mean(psd[(f >= fmin) * (f <= fmax)])
    return psd


def create_groups(y):
    """Generate groups from labels of shape (subject x labels)."""
    k = 0
    y = np.asarray(y)
    groups = []
    for sub in y:
        for _ in range(len(sub)):
            groups.append(k)
        k += 1
    groups = np.asarray(groups).ravel()
    y = np.concatenate(y[:], axis=0).ravel()
    return y, groups


def elapsed_time(t0, t1, formating=True):
    """Time lapsed between t0 and t1.

    Returns the time (from time.time()) between t0 and t1 in a
    more readable fashion.

    Parameters
    ----------
    t0: float,
        time.time() initial measure of time
        (eg. at the begining of the script)
    t1: float,
        time.time() time at the end of the script
        or the execution of a function.

    """
    lapsed = abs(t1 - t0)
    if formating:
        m, h, j = 60, 3600, 24 * 3600
        nbj = lapsed // j
        nbh = (lapsed - j * nbj) // h
        nbm = (lapsed - j * nbj - h * nbh) // m
        nbs = lapsed - j * nbj - h * nbh - m * nbm
        if lapsed > j:
            formated_time = "{:.0f}j, {:.0f}h:{:.0f}m:{:.0f}s".format(
                nbj, nbh, nbm, nbs
            )
        elif lapsed > h:
            formated_time = "{:.0f}h:{:.0f}m:{:.0f}s".format(nbh, nbm, nbs)
        elif lapsed > m:
            formated_time = "{:.0f}m:{:.0f}s".format(nbm, nbs)
        else
            formated_time = "{:.4f}s".format(nbs)
        return formated_time
    return lapsed


def prepare_data(dico, key="data", n_trials=None, random_state=None):
    data = dico[key].ravel()
    final_data = None
    for submat in data:
        if submat.shape[0] == 1:
            submat = submat.ravel()
        if n_trials is not None:
            index = np.random.RandomState(random_state).choice(
                range(len(submat)), n_trials, replace=False
            )
            prep_submat = submat[index]
        else:
            prep_submat = submat
        final_data = (
            prep_submat
            if final_data is None
            else np.concatenate((prep_submat, final_data))
        )

    return np.asarray(final_data)


def load_hypno(sub):
    HYPNO_PATH = path(
        "/home/arthur/Documents/data/sleep_data/sleep_raw_data/hypnograms"
    )
    with open(HYPNO_PATH / "hyp_per_s{}.txt".format(sub)) as f:
        hypno = []
        for line in f:
            if line[0] not in ["-", "\n"]:
                hypno.append(line[0])
    return hypno


# def visu_hypno(sub):
# hypno = list(map(int, load_hypno(sub)))
# plt.plot(hypno)
# plt.show()


def empty_stage_dict():
    stages = ["S1", "S2", "S3", "S4", "REM"]
    stage_dict = {}
    for st in stages:
        stage_dict[st] = []
    return dict(stage_dict)


def split_cycles(data, sub, duree=1200):
    stages = ["S1", "S2", "S3", "S4", "REM"]
    ref = "12345"
    cycles = [empty_stage_dict()]
    hypno = load_hypno(sub)
    for i, hyp in enumerate(hypno):
        next_hyps = hypno[i + 1 : i + 1 + duree]
        obs = data[i * 1000 : (i + 1) * 1000]
        if hyp in ref:
            cycles[-1][stages[ref.index(hyp)]].append(obs)
            if hyp == "5" and "5" not in next_hyps and len(cycles[-1]["REM"]) >= 300:
                cycles.append(dict(empty_stage_dict()))
    return cycles


def convert_sleep_data(data_path, sub_i, elec=None):
    """Load the samples of a subject for a sleepstate."""
    tempFileName = data_path / "s%i_sleep.mat" % (sub_i)
    try:
        if elec is None:
            dataset = np.asarray(h5py.File(tempFileName, "r")["m_data"])[:, :19]
        else:
            dataset = np.asarray(h5py.File(tempFileName, "r")["m_data"])[:, elec]
    except (IOError):
        print(tempFileName, "not found")
    cycles = split_cycles(dataset, sub_i)
    dataset = []
    for i, cycle in enumerate(cycles):
        for stage, secs in cycle.items():
            if len(secs) != 0:
                secs = np.array(secs)
                save = np.concatenate(
                    [secs[i : i + 30] for i in range(0, len(secs), 30)]
                )
                savemat(data_path / "{}_s{}_cycle{}".format(stage, sub_i, i+1), {stage: save})


def merge_S3_S4(data_path, sub_i, cycle):
    try:
        S3_file = data_path / "S3_s{}_cycle{}.mat".format(sub_i, cycle)
        S3 = loadmat(S3_file)["S3"]
        S4_file = data_path / "S4_s{}_cycle{}.mat".format(sub_i, cycle)
        S4 = loadmat(S4_file)["S4"]
        data = {"SWS": np.concatenate((S3, S4), axis=0)}
        savemat(data_path / "SWS_s{}_cycle{}.mat".format(sub_i, cycle), data)
        S3_file.remove()
        S4_file.remove()
    except IOError:
        print("file not found for cycle", cycle)


def merge_SWS(data_path, sub_i, cycle=None):
    if cycle is None:
        for i in range(1, 4):
            merge_S3_S4(data_path, sub_i, i)
    else:
        merge_S3_S4(data_path, sub_i, cycle)


def load_full_sleep(data_path, sub_i, state, cycle=None):
    """Load the samples of a subject for a sleepstate."""
    tempFileName = data_path / "{}_s{}.mat".format(state, sub_i)
    if cycle is not None:
        tempFileName = data_path / "{}_s{}_cycle{}.mat".format(state, sub_i, cycle)
    try:
        dataset = loadmat(tempFileName)[state]
    except (IOError, TypeError) as e:
        print(tempFileName, "not found")
        dataset = None
    return dataset


def load_samples(data_path, sub_i, state, cycle=None, elec=None):
    """Load the samples of a subject for a sleepstate."""
    if elec is None:
        dataset = load_full_sleep(data_path, sub_i, state, cycle)[:19]
    else:
        dataset = load_full_sleep(data_path, sub_i, state, cycle)[elec]
    dataset = dataset.swapaxes(0, 2)
    dataset = dataset.swapaxes(1, 2)
    return dataset


def import_data(data_path, state, subject_list, label_path=None, full_trial=False):
    """Transform the data and generate labels.

    Takes the original files and put them in a matrix of
    shape (Trials x 19 x 30000)

    """
    X = []
    print("Loading data...")
    for i in range(len(subject_list)):
        # Loading of the trials of the selected sleepstate
        dataset = load_samples(data_path, subject_list[i], state)

        if full_trial:
            # use if you want full trial
            dataset = np.concatenate((dataset[range(len(dataset))]), axis=1)
            dataset = dataset.reshape(1, dataset.shape[0], dataset.shape[1])
        X.append(dataset)

        del dataset

    if label_path is not None:
        y = loadmat(label_path / state + "_labels.mat")["y"].ravel()
    return X, np.asarray(y)


def is_signif(pvalue, p=0.05):
    """Tell if condition with classifier is significative.

    Returns a boolean : True if the condition is significativeat given p
    """
    answer = False
    if pvalue <= p:
        answer = True
    return answer


class BaseStratCrossValidator(with_metaclass(ABCMeta)):
    """Base class for all cross-validators.

    Implementations must define `_iter_test_masks` or `_iter_test_indices`.
    """

    def __init__(self):
        # We need this for the build_repr to work properly in py2.7
        # see #6304
        pass

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, of length n_samples
            The target variable for supervised learning problems.
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Returns
        -------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        indices = np.arange(_num_samples(X))
        for test_index in self._iter_test_masks(X, y, groups):
            train_index = indices[np.logical_not(test_index)]
            test_index = indices[test_index]
            if y[test_index[0]] != y[test_index[-1]]:
                yield train_index, test_index

    # Since subclasses must implement either _iter_test_masks or
    # _iter_test_indices, neither can be abstract.
    def _iter_test_masks(self, X=None, y=None, groups=None):
        """Generates boolean masks corresponding to test sets.

        By default, delegates to _iter_test_indices(X, y, groups)
        """
        for test_index in self._iter_test_indices(X, y, groups):
            test_mask = np.zeros(_num_samples(X), dtype=np.bool)
            test_mask[test_index] = True
            yield test_mask

    def _iter_test_indices(self, X=None, y=None, groups=None):
        """Generates integer indices corresponding to test sets."""
        raise NotImplementedError

    @abstractmethod
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator."""

    def __repr__(self):
        return _build_repr(self)


class StratifiedLeavePGroupsOut(BaseStratCrossValidator):
    """Leave P Group(s) Out cross-validator.

    Only stratified for n_groups = 2 !!!!!
    Provides train/test indices to split data according to a third-party
    provided group. This group information can be used to encode arbitrary
    domain specific stratifications of the samples as integers.
    For instance the groups could be the year of collection of the samples
    and thus allow for cross-validation against time-based splits.
    The difference between LeavePGroupsOut and LeaveOneGroupOut is that
    the former builds the test sets with all the samples assigned to
    ``p`` different values of the groups while the latter uses samples
    all assigned the same groups.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_groups : int
        Number of groups (``p``) to leave out in the test split.
    Examples
    --------
    >>> from sklearn.model_selection import LeavePGroupsOut
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> y = np.array([1, 2, 1])
    >>> groups = np.array([1, 2, 3])
    >>> lpgo = LeavePGroupsOut(n_groups=2)
    >>> lpgo.get_n_splits(X, y, groups)
    3
    >>> print(lpgo)
    LeavePGroupsOut(n_groups=2)
    >>> for train_index, test_index in lpgo.split(X, y, groups):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    ...    print(X_train, X_test, y_train, y_test)
    TRAIN: [2] TEST: [0 1]
    [[5 6]] [[1 2]
     [3 4]] [1] [1 2]
    TRAIN: [1] TEST: [0 2]
    [[3 4]] [[1 2]
     [5 6]] [2] [1 1]
    TRAIN: [0] TEST: [1 2]
    [[1 2]] [[3 4]
     [5 6]] [1] [2 1]
    See also
    --------
    GroupKFold: K-fold iterator variant with non-overlapping groups.
    """

    def __init__(self, n_groups):
        self.n_groups = n_groups

    def _iter_test_masks(self, X, y, groups):
        if groups is None:
            raise ValueError("The groups parameter should not be None")
        groups = check_array(groups, copy=True, ensure_2d=False, dtype=None)
        unique_groups, group_counts = np.unique(groups, return_counts=True)
        if self.n_groups >= len(unique_groups):
            raise ValueError(
                "The groups parameter contains fewer than (or equal to) "
                "n_groups (%d) numbers of unique groups (%s). LeavePGroupsOut "
                "expects that at least n_groups + 1 (%d) unique groups be "
                "present" % (self.n_groups, unique_groups, self.n_groups + 1)
            )
        # unique_groups = np.delete(unique_groups,
        #                           np.where(group_counts < len(X)/(2*len(unique_groups))))
        combi = combinations(range(len(unique_groups)), self.n_groups)
        for indices in combi:
            test_index = np.zeros(_num_samples(X), dtype=np.bool)
            for l in unique_groups[np.array(indices)]:
                test_index[groups == l] = True

            if len(test_index) > len(X) / (2 * len(unique_groups)):
                yield test_index

    # Unused not working
    def get_n_splits(self, X, y, groups):
        """Returns the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.
            ``np.zeros(n_samples)`` may be used as a placeholder.
        y : object
            Always ignored, exists for compatibility.
            ``np.zeros(n_samples)`` may be used as a placeholder.
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        if groups is None:
            raise ValueError("The groups parameter should not be None")
        groups = check_array(groups, ensure_2d=False, dtype=None)
        X, y, groups = indexable(X, y, groups)
        return int(comb(len(np.unique(groups)), self.n_groups, exact=True))


# Unused not working
def _build_repr(self):
    # XXX This is copied from BaseEstimator's get_params
    cls = self.__class__
    init = getattr(cls.__init__, "deprecated_original", cls.__init__)
    # Ignore varargs, kw and default values and pop self
    init_signature = signature(init)
    # Consider the constructor parameters excluding 'self'
    if init is object.__init__:
        args = []
    else:
        args = sorted(
            [
                p.name
                for p in init_signature.parameters.values()
                if p.name != "self" and p.kind != p.VAR_KEYWORD
            ]
        )
    class_name = self.__class__.__name__
    params = dict()
    for key in args:
        # We need deprecation warnings to always be on in order to
        # catch deprecated param values.
        # This is set in utils/__init__.py but it gets overwritten
        # when running under python3 somehow.
        warnings.simplefilter("always", DeprecationWarning)
        try:
            with warnings.catch_warnings(record=True) as w:
                value = getattr(self, key, None)
            if len(w) and w[0].category == DeprecationWarning:
                # if the parameter is deprecated, don't show it
                continue
        finally:
            warnings.filters.pop(0)
        params[key] = value

    return "%s(%s)" % (class_name, _pprint(params, offset=len(class_name)))


def StratifiedLeave2GroupsOut():
    return StratifiedLeavePGroupsOut(n_groups=2)
