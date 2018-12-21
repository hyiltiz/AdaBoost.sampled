"""AdaBoost binary classifier using sampled boosting stumps. You can feed in
libsvm formatted text files for binary classification. Creating training and
test data sets, generating result plots and tables and algorithm analysis are
all performed automatically. You can also do the above with cross validation.
This module is designed so that it can be imported to work with your own
program. Functional programming principles are followed whenever possible.

Usage:
  adaboost.py run [-i <dataset>] [-r <sampleClassifierRatio>] [-T <T>] [--seed <seed>] [--log <loglevel>]
  adaboost.py convert [-i <dataset>] [-s <s>]
  adaboost.py cv [-i <dataset>] [-k <k>] [-r <sampleClassifierRatio>] [-T <T>] [--seed <seed>] 
  adaboost.py (-h | --help)
  adaboost.py (--version)

Options:
  -h --help                          Show this screen.
  --version                          Show version.
  -i <dataset>                       Data set to perform adaBoost. [default: breast-cancer.txt]
  -r <sampleClassifierRatio>         Percentage of classifiers used. [default: 1.0]
  --seed <seed>                      Seed to initialize adaboost() function. Controls sampler. [default: 0]
  --loglevel <loglevel>              Use INFO to record classifiers evaluated each round. [default: NOTSET]
  -k <k>                             k-fold cross validation. [default: 10]
  -s <s>                             Split s percent of data into training set. [default: 0.8]
  -T <T>                             Number of rounds to boost. [default: 1e3]

"""

from docopt import docopt
import sys
import os.path
import os
import datetime
import logging
import multiprocessing # noqa: TODO
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rc


#rc('font', **{'family': 'serif',
#              'serif': ['Palatino'],
#              'sans-serif': ['Helvetica']})
rc('text', usetex=True)

# use this to debug
# import pdb; pdb.set_trace()  # noqa


def adaBoost(data_npy='breast-cancer', sampleRatio=(0, 0), T=int(1e2),
             seed=0, loglevel=0):
    """Perform sampled classifier AdaBoost (AdaBoost.SC) with boosting stumps
    (threshold functions) as base classifiers. With m samples, N features and T
    iterations of boosting, the total computational complexity is O(mNlogm +
    mNT).

    @ Parameters:
    data_npy: It could be one of the following:
       1. a string for files <data_npy>_train0.py and <data_npy>_test0.py.
       2. a string as the name of a libsvm formatted text file.
       3. a 3-tuple containing two numpy arrays of shape (m, N) and a string
          for the name of the data set as an identifier (trainArray, testArray,
          idStr). This format is particularly useful for cross-validation or
          other tests where data are transformed before analysis dynamically.

    sampleRatio: a tuple (a,b) where positive ratios a, b represents the
    percentage of samples and base classifiers to be used to compute errors
    during each round.

    T: number of iterations. Note that the empirical error of the learned
       classifier g: $\hat{R}(g) \leq e^{-2 \sum_{t=1}^T (\gamma_t)^2}$ where
       \gamma_t is the edge of the base classifier picked at round t.

    seed: seed used to initialize the algorithm. Used for picking classifiers
          randomly at each round.

    loglevel: numeric value. Use 0 to disable logs, and 20 to enable logs.
    Currently, the logs include the gammas of all classifiers evaluated during
    each round.

    Takes a numpy file path for data storing a numpy array and a function that
    generates base classifiers, evaluates all base classifiers when isSample is
    False, and runs classic AdaBoost.

    @ Returns: learned ensemble g as a list of 3-tuples of thresholds,
              and the corresponding features and direction, and weights.
              Feed g into predict() to generate predictions and errors.

    In addition, it creates a pdf containing plots of error distributions of
    the base classifiers and the classifiers in the ensemble, and training and
    test errors as a function of round t.

    # TODO: A re-implementation to generalize base classifiers [(ID, alpha)] is
            the base classifiers, linked to an evaluation function (ID,x) -> y.
            We construct a base classifier bank [(ID, alpha), h_i_t,
            errors_train] upon creation. Then base classifiers are only
            evaluated once.

    """
    sampleDataRatio, sampleClassifierRatio = sampleRatio


    data_npy, data, data_test = getData(data_npy)

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    pp = PdfPages(data_npy + '_results_plots_' + timestamp + '.pdf')

    logFilename = '{}_classifiers_history_seed{}_sampleRatio{}_{}.log'.format(
        data_npy, seed, sampleRatio[1], timestamp)
    if loglevel > 0:
        logging.basicConfig(format='%(message)s',
                            filename=logFilename,
                            filemode='w',
                            level=loglevel)
        logging.info('weighted_error, threshold, feature_direction, iteration')

    m_samples = data.shape[0]
    stumps = createBoostingStumps(data, loglevel)
    nStumps = len(stumps)
    print('Number of base classifiers: {}'.format(nStumps))

    # write into a table and also generate a histogram of errors
    stumpsFig = writeStumps2CSV(stumps, data_npy + '_stumps')
    pp.savefig(stumpsFig)

    # some auxiliary variables
    auxVars = {'t': -1,
               'nStumps': nStumps,
               'loglevel': loglevel}

    # AdaBoost algorithm
    D_t = np.zeros(m_samples)+1.0/m_samples
    h = []
    alpha = np.zeros(T)
    np.random.seed(seed)
    for t in tqdm(range(0, T)):
        auxVars['t'] = t
        e_t, h_t = evalToPickClassifier(stumps, D_t, data, sampleRatio,
                                                **auxVars)
        h.append(h_t)  # keep the errors
        alpha[t] = 1.0/2 * np.log(1/e_t-1)
        # not keeping track of D_t, Z_t history to optimize for memory
        Z_t = 2*np.sqrt(e_t*(1-e_t))
        # Note that we have: errors === (-y * h_i_x+1)/2
        D_t = D_t * np.exp(alpha[t]*(2*h_t[2]-1))/Z_t

    # Construct the ensemble out of the picked classifiers, with alphas
    g = [(hi[1][0], hi[1][1], alpha[i]) for i, hi in enumerate(h)]

    # Save the results to a csv table and generate some plots
    auxVars2 = {'logFilename': logFilename,
                'sampleClassifierRatio': sampleRatio[1], 
                'gammaHistoryFile': '{}_ensemble_history_seed{}_sampleRatio{}'
                '_{}.csv'.format(data_npy, seed, sampleRatio[1],
                                 timestamp)}

    pp, error_history, logHistory = generateResults(
        g, h, (data_npy, data, data_test), pp,
        **auxVars2)
    pp.close()

    # results saved, now return the ensemble
    # and training errors of each classifier in the ensemble
    return g, h, error_history


def createBoostingStumps(data, loglevel):
    """
    Create boosting stumps, i.e. axis aligned thresholds for each features.
    Sorts the data first and uses the sorted values for each component to
    construct the thresholds. Has complexity O(mNlogm).
    """

    # we use python lists with tuples as they are actually faster in our case
    # that numpy arrays
    baseClassifiers = []
    # NOTE: these loops can run in parallel
    for iFeature in range(1, data.shape[1]):  # 0th column is the label
        thresholds = np.unique(data[:, iFeature])
        for iThreshold in thresholds:
            baseClassifiers.append(applyStump2Data(
                [], data, evals=True,
                **{'iThreshold': iThreshold,
                   'iFeature': iFeature,
                   'D_t': None,
                   'alpha': None,
                   'loglevel': loglevel}))

    return baseClassifiers


def evalToPickClassifier(stumps, D_t, data, sampleRatio, t, nStumps, loglevel):
    """Pick a classifier from a `sampleRatio` percent of the base classifiers
    `stumps` that has the smallest error w.r.t. the weights D_t on the m data
    points. This function actually do *not* evaluate the classifiers. Rather,
    errors for each base classifiers were stored in `stumps` for computational
    efficiency, so this function merely samples base classifiers and re-weights
    the errors to pick the best base classifiers w.r.t. the new weight D_t.

    NOTE: This function currently only samples the classifiers.

    """

    sampleDataRatio, sampleClassifierRatio = sampleRatio

    # TODO: sampling data is not implemented yet
    # Treat these the same: no sampling or sample everything
    if sampleDataRatio == 0:
        sampleDataRatio == 1
    index_data = np.random.rand(1, data.shape[1]-1) < sampleDataRatio  # noqa

    # treat these the same: no sampling or sample everything
    if sampleClassifierRatio == 0:
        sampleClassifierRatio = 1
    index_classifiers = np.random.rand(1, nStumps) < sampleClassifierRatio
    index_classifiers_list = np.ndarray.tolist(np.where(index_classifiers)[1])

    # We kept the classification errors in `stumps` for each base classifier
    # upon creation therefore we no longer need to actually evaluate them and
    # compare with class labels y to compute errors for computational
    # efficiency sacrificing some RAM. y = data[:, 0] D_t = 1.0/data.shape[0]

    # Evaluate a subset of classifiers keeping track of the smallest error
    # We keep a list of best in samples, not a tuple so we could check what
    # was the second best, or our worst performance later
    sampleErrors = np.zeros(index_classifiers.sum())
    # NOTE: these loops can run in parallel
    for _, iStump in enumerate(index_classifiers_list):
        # load a classifier
        stumps[iStump] = applyStump2Data(  # only change some stumps
            stumps[iStump], data, evals=False,
            **{'D_t': D_t,
               'iThreshold': None,
                'iFeature': None,
                'alpha': None,
                'loglevel': loglevel})

    sampleErrors = np.array([stumps[i][0] for i in index_classifiers_list])

    minError = sampleErrors.min()
    idxBestInSample = index_classifiers_list[sampleErrors.argmin()]
    bestInSample = stumps[idxBestInSample]
    return minError, bestInSample


def predict(learnedClassifiers, test_data_npy='breast-cancer_test0.npy'):
    """Use the learned stumps (thresholds for features and their directions) to
    predict labels for new data.

    @ Parameters:
    learnedClassifiers: a list of 3-tuples (threshold, feature*direction,
      alpha) representing an ensemble of boosting stumps.

    @ Returns: classification error, predicted labels and errors
    """

    if type(test_data_npy) is str:
        data = np.load(test_data_npy)
    elif type(test_data_npy) is np.ndarray:
        data = test_data_npy

    evaluatedClassifiers = []
    for stump in learnedClassifiers:  # 0th column is the label
        evaluatedClassifiers.append(applyStump2Data(
            (None, stump, None, None), data, evals=True,
            **{'iThreshold': None,
               'D_t': None,
               'iFeature': None,
               'alpha': None,
               'loglevel': 0}))

    h_x = np.zeros((data.shape[0], len(learnedClassifiers)))
    for iStump, stump in enumerate(evaluatedClassifiers):
        h_x[:, iStump] = stump[1][2] * stump[3]   # weighted: alpha*h_i_x

    y = data[:, 0]
    y_predict = np.sign(np.sum(h_x, 1))           # majority

    errors = ((y != y_predict)+0.0)*2-1
    error = np.sum((y != y_predict)+0.0)/y.shape[0]

    return error, y_predict, y, evaluatedClassifiers


def applyStump2Data(stump, data, evals,
                    D_t, iThreshold, iFeature, alpha, loglevel):
    # locals().update(kwargs)  # HACK: load kwargs into function namespace
    if iThreshold is None:
        # Just unpack stump
        iThreshold, temp, alpha = stump[1]
        iDirection = np.sign(temp)
        iFeature = np.abs(temp)
        h_i_x = stump[3]
        errors = stump[2]

    if evals is True:
        # need iFeature, iThreshold
        y = data[:, 0]
        iDirection = +1
        # Here D_t is uniform to create the stumps
        D_t = 1.0/y.size
        # NOTE: This is the only place any stump is actually evaluated
        h_i_x = ((data[:, iFeature] >= iThreshold)+0)*2-1  # {-1, 1} labels
        errors = (-y * h_i_x+1)/2  # {0, 1}, 1 records misclassified data

    # Invert the classifier if its error exceeds random guessing
    # Here D_t could change between rounds
    weighted_error = np.sum(D_t * errors)
    if weighted_error > 0.5:
        iDirection = -iDirection
        h_i_x = -h_i_x
        errors = 1-errors
    weighted_error = np.sum(D_t * errors)

    # record this classifier
    # alpha, to be used by predict() after adaBoost() finishes training
    # $\epsilon_t$: weighted_error weights classification errors by D_t
    newStump = (
        weighted_error,  # used to pick classifiers and analyze algorithm
        (iThreshold, iDirection*iFeature, alpha),  # the classifier tuple
        errors,          # classifier errors, stored to prevent evaluation
        h_i_x)           # classifier evaluation, also stored

    if loglevel > 0:
        logging.info('{}, {}, {}, {}'.format(
            weighted_error, iThreshold, iDirection*iFeature, 0))
    return newStump


def writeStumps2CSV(stumps, fname):
    """Write a list of classifiers `stumps` to a CSV file `fname`, and returns a
    handle to a histogram of classifiers edges (classifier accuracy above
    chance).

    """

    nStumps = len(stumps)
    stumpsTable = np.zeros((nStumps, 4))
    for iStump in range(nStumps):
            stumpsTable[iStump, :] = np.hstack((
                    np.array(stumps[iStump][0]),  # weighted_error
                    np.array(stumps[iStump][1])   # stumps, a 3-tuple
            ))

    np.savetxt(fname + '.dump.csv', stumpsTable,
               delimiter=',', newline='\n', comments='',
               header='weighted_error, threshold, feature, weight_alpha')

    fig = plt.figure()
    # 0.5 - weighted_error == gamma only when D_t is uniform, i.e. for base
    # classifiers `stumps`.
    # This does not hold for g, i.e. post training ensemble classifier.
    plt.hist(0.5 - stumpsTable[:, 0])
    stumpsOrEnsembleStr = fname.split('_')[-1]
    plt.title('Distribution of edge for ' + stumpsOrEnsembleStr)
    plt.xlabel('Classifier edge $\gamma_i$')
    plt.ylabel('$Frequency$')
    return fig


def generateResults(g, h, dataTuple, pp,
                    gammaHistoryFile, logFilename, sampleClassifierRatio):
    """Create result plots and tables using the ensemble g (includes alpha) and h
    (includes errors).
    Created tables:
    Created plots:
      1. Histogram of edges of classifiers in the ensemble
      2. Test and training error of the ensemble as a function of iterations t
         for <data_npy>_train0.npy and <data_npy>_test0.npy.
      3. Distribution of errors for all evaluated base classifiers at each
         round. This needs the logs to be enabled, and is computationally
         expensive to plot and render.

    """
    data_npy, data, data_test = dataTuple
    output = predict(g, data)
    h.append((output[0], (-999, -999, -999)))
    ensembleFig = writeStumps2CSV(h, data_npy + '_ensemble')
    h.pop()  # already recorded the ensemble error in CSV
    pp.savefig(ensembleFig)

    # Compute error history as the ensemble grows
    # Instead of actually growing the ensemble, we use matrix tricks
    _, _, y_test, h_test = predict(g, data_test)
    T = len(g)
    m = len(h[0][2])
    m_test = len(h_test[0][2])
    h_i_x_train = np.zeros((m, T))
    h_i_x_test = np.zeros((m_test, T))
    weighted_error_train = np.zeros(T)
    alpha = np.zeros(T)
    for hi in range(0, T):
        h_i_x_train[:, hi] = h[hi][3]
        weighted_error_train[hi] = h[hi][0]  # to bound empirical (test) error
        h_i_x_test[:, hi] = h_test[hi][3]
        alpha[hi] = g[hi][2]

    Y = np.tile(data[:, 0], (T, 1)).T
    y_predict_train_history = np.matmul(h_i_x_train, np.tril(alpha).T)
    errors_train_history = ((((y_predict_train_history > 0)+0)*2-1) != Y)+0
    error_train_history = (errors_train_history.sum(0)+0.0)/Y.shape[0]

    Y_test = np.tile(y_test, (T, 1)).T
    y_predict_test_history = np.matmul(h_i_x_test, np.tril(alpha).T)
    errors_test_history = ((((y_predict_test_history > 0)+0)*2-1) != Y_test)+0
    error_test_history = (errors_test_history.sum(0)+0.0)/Y_test.shape[0]

    e = np.tril(weighted_error_train).T  # See Mohri (2012) pp. 125
    Z = 2*np.sqrt(e*(1-e))
    edges = np.tril(0.5-weighted_error_train).T
    Z[Z==0] = 1
    empiricalError = Z.prod(0)
    empiricalErrorBound63 = np.exp(-2*np.sum(np.power(edges, 2), 0))
    edges[edges==0] = 0.5  # rewrite 0 into 0.5 so argmin works
    empiricalErrorBound64 = np.exp(-2*np.power(edges.min(0), 2)*range(1,T+1))
    sigma = ((np.random.rand(int(1e5), m) < 0.5)+0)*2-1  # Rademacher variable
    empiricalRademacher = np.max(np.matmul(sigma, h_i_x_train)/m, 1).mean()
    rho = 0.01
    deltaConfidence = 0.05
    Y_predict = (((y_predict_train_history>0)+0)*2-1)
    marginLosses, onMargin = np.vectorize(fMargin)((Y*Y_predict)/np.tril(alpha).T.sum(0), rho)
    marginLoss = marginLosses.mean(0)
    generalizationBound616 = marginLoss + 2/rho*empiricalRademacher + \
        3*np.sqrt(np.log(2.0/deltaConfidence)/(2*m))  # Mohri (2012) pp. 133
    # NOTE: this bound is larger than 1 for breast-canter 3% classifiers with
    # a margin of 0.01 with 95% confidence

    error_history = np.vstack((
        range(T),
        error_train_history,
        error_test_history,
        empiricalErrorBound63,
        empiricalErrorBound64,
        generalizationBound616)).T

    print('The test error for {} using base classifiers with a (empirical) '
          'Rademacher complexity of {} was: {}'.format(
              data_npy+'_test0.npy', empiricalRademacher,
              error_test_history[-1]))
    np.savetxt(gammaHistoryFile, error_history,
               delimiter=',', comments='',
               header='iteration, train-error, test-error, '
               'empirical-error-bound63, empirical-error-bound64, '
               'generalization-bound616')

    # Create a plot for error history over iterations.
    historyFig = plt.figure()
    plt.plot(error_history[:, 0], error_history[:, 1], label='train')
    plt.plot(error_history[:, 0], error_history[:, 2], label='test')
    plt.legend()
    plt.title('Ensemble error using ({:g}\% of stumps)'.format(
        sampleClassifierRatio*100))
    plt.xlabel('Iteration $t$')
    plt.ylabel('Error $\epsilon_t$')
    pp.savefig(historyFig)

    # If logs were enabled during iterations, then generate plots for it too
    # Plot erros history histribution of all evaluated classifiers during each
    # round
    logHistory = np.zeros(1)
    if (os.path.isfile(logFilename) and
            len(open(logFilename, 'rb').readlines()) >= 2):
        logHistory = np.loadtxt(logFilename, delimiter=',', skiprows=1)
        historyErrorsEvaluated = plt.figure()
        plt.scatter(logHistory[:, 3], logHistory[:, 0], s=0.1)
        plt.xlabel('Iterations $t$')
        plt.ylabel('Evaluated classifier error $\epsilon_{t,i}$')
        plt.title('Training history using ({:g}\% of stumps)'.format(
            sampleClassifierRatio*100))
        pp.savefig(historyErrorsEvaluated)

    return pp, error_history, logHistory


def fMargin(x, rho):
    """Margin loss function with a margin rho"""
    if x <= 0:
        return 0, 0
    elif x >= rho:
        return 1, 0
    else:
        return 1.0 - x/rho, 1


def libsvmLineParser(line):
    """Parses a libsvm data line to a dictionary.

    NOTE: libsvm data files for binary classification are .txt files where each
lines starts with an integer representing class labels, followed by several n:f
separated by spaces where n is an integer representing the nth feature with
value f.
    # TODO: Extend it to deal with multilabel classification parsing.

    """

    D = dict()
    columns = line.strip().split(' ')
    try:
        D = dict((int(k), float(v))
                 for k, v in (e.split(':')
                              for e in columns[1:]))
        D[0] = float(columns[0])

    except ValueError:
        print('Could not parse the following line:')
        print(line)

    return D


def libsvmReadTxt(file):
    """Reads a libsvm formatted .txt file into a numpy array. The cases or features
that has no data (complete missing value rows or columns) are pruned out, but
sparse missing values were kept intact.

    """

    fh = open(file)
    lines = fh.read().split('\n')
    fh.close()

    # Parse lines into a list of dictionaries
    # [['string']] -> [dict()]
    listD = []
    for line in lines:
        if line is not '':
            listD.append(libsvmLineParser(line))

    # Convert the list of dictionary into an array
    # [dict()] -> np.array()
    nFeatures = max([max(i) for i in listD]) + 1  # Feature index from 0
    mSamples = len(listD)
    data = np.zeros((mSamples, nFeatures)) + np.nan
    for i, iSample, in enumerate(listD):
        for k, v in iSample.iteritems():
            data[i, k] = v

    # remove fully missing values
    missingCases = np.isnan(data).all(1)
    missingFeatures = np.isnan(data).all(0)
    if missingCases.nonzero()[0].size > 0:
        print('Removing completely {} missing cases.'.format(
            missingCases.nonzero()[0].size))
        data = data[~missingCases, :]
    if missingFeatures.nonzero()[0].size:
        print('Removing completely {} missing features.'.format(
            missingFeatures.nonzero()[0].size))
        data = data[:, ~missingFeatures]
    if np.isnan(data).any():
        print('There are still {} missing data points.'.format(
            np.isnan(data).nonzero()[0].size))
        # TODO: only remove cases or features s.t. least data points are
        # removed as well to preserve as much data as possible
        # HACK: remove all missing cases for now
        data = data[~np.isnan(data).any(1), :]

    # normalize the data into [-1,1]
    scale = lambda x: (x-np.min(x,0))/(np.max(x,0)-np.min(x,0))*2-1

    # Do not scale class labels
    #   datarray = np.column_stack([data[:,0], scale(data[:,1:])])

    # scale class labels as well
    # breast-cancer has labels of 2, 4

    data = scale(data)
    return data


def analyzeCVError(data_txt, k=10, sampleClassifierRatio=1.0, T=1e4, seed=0):
    """k-fold cross validation error of the ensemble over time."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    loglevel = 0
    data_npy, data_list, data_test_list = getData(data_txt, k)
    error_history_cv = np.array([])
    for data, data_test in zip(data_list, data_test_list):
        _, _, error_history = adaBoost(
            (data_npy, data, data_test),
            (0, sampleClassifierRatio), T, seed, loglevel)

        error_history_cv = np.dstack((
            error_history_cv,
            error_history)) if error_history_cv.size else error_history

    CVError = (error_history_cv.mean(2), error_history_cv.std(2)/np.sqrt(k))

    # Create a plot for error history over iterations.
    error_history, error_history_sem = CVError
    legends = ['iteration', 'train-error', 'test-error',
               'empirical-error-bound63', 'empirical-error-bound64',
               'generalization-bound616']
    historyFig = plt.figure()
    for i in range(1, 4):
        plt.errorbar(range(T), error_history[:, i], label=legends[i])

    plt.legend()
    plt.title('Ensemble 10-fold-CV error using ({:g}\% of stumps)'.format(
        sampleClassifierRatio*100))
    plt.xlabel('Iteration $t$')
    plt.ylabel('Error $\epsilon_t$')
    plt.savefig('{}_{}CV_seed{}_sampleRatio{}_{}.eps'.format(
        data_npy, k, seed, sampleClassifierRatio, timestamp),
                dpi = 1200, format = 'eps')
    plt.ylim((0,1))

    return CVError


def getData(data_npy, k=0.8):
    """Reads data_npy into a string indicating the data set, numpy arrays of
training and test data sets. If k>1 and the input is a libsvm .txt file, split
into k training and test sets for cross-validation. If 0<k<1, then the data is
split randomly where k percent is used for training and 1-k for testing.

    """

    if type(data_npy) is tuple:
        # do nothing
        data_npy, data, data_test = data_npy

    elif type(data_npy) is str:  # noqa
        try:
            data = np.load(data_npy + '_train0.npy')
            data_test = np.load(data_npy + '_test0.npy')
            data = np.load(os.getcwd() + '/' + data_npy + '_train0.npy')
        except IOError:
            print('Failed to load {} as .npy file. '
                  'Assuming it is a libsvm .txt file and '
                  'converting to a numpy array.'.format(data_npy))
            data_full = libsvmReadTxt(data_npy)
            if k < 1:
                # split into test and training set by 20%
                idxTrain = np.random.rand(data_full.shape[0]) < k
                data = data_full[idxTrain, :]
                data_test = data_full[~idxTrain, :]
                data_npy = data_npy + '-converted'
            elif k >= 1:
                CVsplits = np.repeat(range(k), data_full.shape[0]/k)
                # remove remainders after folding into chunks
                data_full = data_full[0:len(CVsplits), :]
                data_list = []
                data_test_list = []
                for iSplit in range(k):
                    data_test_list.append(data_full[CVsplits == iSplit, :])
                    data_list.append(data_full[CVsplits != iSplit, :])

                data_npy = data_npy + '-cv'
                # return lists of data
                return data_npy, data_list, data_test_list

    return data_npy, data, data_test


if __name__ == '__main__':
    args = docopt(__doc__, version='Naval Fate 2.0')

    if args['convert']:
        data_npy = args['-i']
        data_full = libsvmReadTxt(data_npy)
        idxTrain = np.random.rand(data_full.shape[0]) < float(args['-s'])
        data = data_full[idxTrain, :]
        data_test = data_full[~idxTrain, :]
        data_npy = data_npy + '-converted'
        np.savetxt(data_npy + '-converted_train0.csv', data)
        np.savetxt(data_npy + '-converted_test0.csv', data_test)
    elif args['cv']:
        print('Performing cross validation.')
        CVError = analyzeCVError(
            args['-i'], int(args['-k']),
            float(args['-r']), int(float((args['-T']))), int(args['--seed']))
    elif args['run']:
        print('Training {} with {}% stumps for {} iterations ...'.format(
            args['-i'], float(args['-r'])*100, int(float(args['-T']))))
        g = adaBoost(args['-i'], (0, float(args['-r'])), int(float((args['-T']))),
                     int(args['--seed']), args['<loglevel>'])
