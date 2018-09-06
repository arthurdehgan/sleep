[200~In [1]: for state in STATE_LIST:
    ...:     for freq in FREQS:
    ...:         pscores = []
    ...:         for file in file_list:
    ...:             if freq in file and state in file:
    ...:                 a = loadmat(file)
    ...:                 pscores.append(a['pscore'].ravel())
    ...:         pscores = np.array(pscores).max(axis=0)
    ...:         for file in file_list:
    ...:             if freq in file and state in file:
    ...:                 a = loadmat(file)
    ...:                 score = a['score'].ravel()
    ...:                 pvalue = .0
    ...:                 for pscore in pscores:
    ...:                     if pscore >= score:
    ...:                         pvalue += float(1/1001)
    ...:                 a['pvalue_ncorr'] = a['pvalue'].ravel()
    ...:                 a['pvalue'] = pvalue
    ...:                 savemat(file, a)
 [201~
