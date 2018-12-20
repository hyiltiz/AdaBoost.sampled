import numpy as np
import sys
import os.path
import os
import datetime
import logging
import multiprocessing

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rc
from tqdm import tqdm


# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font', **{'family': 'serif', 'serif': ['Palatino']})
rc('text', usetex=True)

# use this to debug
# import pdb; pdb.set_trace()  # noqa


def adaBoost(data_npy='breast-cancer', sampleRatio=(0,0), T=int(1e2), seed=0, loglevel=0):
    """Perform sampled classifier AdaBoost (AdaBoost.SC) with boosting stumps
    (threshold functions) as base classifiers. With m samples, N features and T
    iterations of boosting, the total computational complexity is O(mNlogm + mNT).

    @ Parameters:
    data_npy: a string to a file strong a numpy array, or a numpy
    array of shape (m,N).

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

    """

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    pp = PdfPages(data_npy + '_results_plots_' + timestamp + '.pdf')
    # or use data directly
    data = np.load(os.getcwd() + '/' + data_npy + '_train0.npy')
    m_samples = data.shape[0]

    logFilename = '{}_classifiers_history_seed{}_sampleRatio{}_{}.log'.format(
        data_npy, seed, sampleRatio[1], timestamp)
    if loglevel > 0:
        logging.basicConfig(format='%(message)s',
                            filename=logFilename,
                            filemode='w',
                            level=loglevel)
        logging.info('weighted_error, threshold, feature_direction, iteration')

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
        e_t, h_t, errors = evalToPickClassifier(stumps, D_t, data, sampleRatio,
                                                **auxVars)
        h.append(h_t[0:2])  # keep the errors
        alpha[t] = 1.0/2 * np.log(1/e_t-1)
        # not keeping track of D_t, Z_t history to optimize for memory
        Z_t = 2*np.sqrt(e_t*(1-e_t))
        # Note that we have: errors === (-y * h_i_x+1)/2
        D_t = D_t * np.exp(alpha[t]*(2*errors-1))/Z_t

    # Construct the ensemble out of the picked classifiers, with alphas
    g = [(hi[1][0], hi[1][1], alpha[i]) for i, hi in enumerate(h)]

    # Save the results to a csv table and generate some plots
    auxVars2 = {'logFilename': logFilename,
                'gammaHistoryFile': '{}_ensemble_history_seed{}_sampleRatio{}'
                '_{}.csv'.format(data_npy, seed, sampleClassifierRatio,
                                   timestamp)}


    pp, error_history, logHistory = generateResults(g, h, data_npy, pp,
                                                    **auxVars2)
    pp.close()

    # results saved, now return the ensemble
    # and training errors of each classifier in the ensemble
    return g, h[0]

def createBoostingStumps(data, loglevel):
    """
    Create boosting stumps, i.e. axis aligned thresholds for each features.
    Sorts the data first and uses the sorted values for each component to
    construct the thresholds. Has complexity O(mNlogm).
    """

    # we use python lists with tuples as they are actually faster in our case
    # that numpy arrays
    baseClassifiers = []
    y = data[:, 0]
    # Here D_t is uniform to create the stumps
    D_t = 1.0/data.shape[0]
    # NOTE: these loops can run in parallel
    for iFeature in range(1, data.shape[1]):  # 0th column is the label
        thresholds = np.unique(data[:, iFeature])
        for iThreshold in thresholds:
            iDirection = +1

            h_i_x = ((data[:, iFeature] >= iThreshold)+0)*2-1
            errors = (-y * h_i_x+1)/2

            # invert the classifier if its error exceeds random guessing
            weighted_error = np.sum(D_t * errors)
            if weighted_error > 0.5:
                iDirection = -iDirection
                errors = 1-errors
                weighted_error = np.sum(D_t * errors)

            # record this classifier
            # alpha, to be used by predict() after adaBoost() finishes training
            alpha = 1.0  # placeholder
            # $\epsilon_t$: weighted_error weights classification errors by D_t
            baseClassifiers.append((
              weighted_error,  # used to pick classifiers and analyze algorithm
              (iThreshold, iDirection*iFeature, alpha),  # the classifier tuple
              errors))        # classifier errors, stored to prevent evaluation

            if loglevel > 0:
                logging.info('{}, {}, {}, {}'.format(
                    weighted_error, iThreshold, iDirection*iFeature, 0))

    return baseClassifiers


def evalToPickClassifier(stumps, D_t, data, sampleRatio, t, nStumps, loglevel):
    """Pick a classifier from a `sampleRatio` percent of the base classifiers
    `stumps` that has the smallest error w.r.t. the weights D_t on the m data
    points. This function actually do *not* evaluate the classifiers. Rather,
    errors for each base classifiers were stored in `stumps` for computational
    efficiency, so this function merely samples base classifiers and re-weights
    the errors to pick the best base classifiers w.r.t. the new weight D_t.

    This function currently only samples the classifiers.
    TODO: Also sample the data.

    """

    sampleDataRatio, sampleClassifierRatio = sampleRatio

    # TODO: sampling data is not implemented yet
    # treat these the same: no sampling or sample everything
    if sampleDataRatio == 0:
        sampleDataRatio == 1
    index_data = np.random.rand(1, data.shape[1]-1) < sampleDataRatio

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
    # TODO: keep a list of best in samples, not a tuple so we could check what
    # was the second best, or our worst performance later
    bestInSample = (-1, 1)
    # NOTE: these loops can run in parallel
    for iStump in index_classifiers_list:            # 0th column is the label
        # load a classifier
        iThreshold, temp, alpha = stumps[iStump][1]  # alpha is not used here
        iFeature = np.abs(temp)
        iDirection = np.sign(temp)
        errors = stumps[iStump][2]

        # invert the classifier if its error exceeds random guessing
        weighted_error = np.sum(D_t * errors)
        if weighted_error > 0.5:
            iDirection = -iDirection
            errors = 1 - errors
            weighted_error = np.sum(D_t * errors)

        if weighted_error < bestInSample[1]:
            bestInSample = (iStump, weighted_error)

        # update this classifier
        alpha = 1.0
        stumps[iStump] = (
            weighted_error,  # used to pick classifiers and analyze algorithm
            (iThreshold, iDirection*iFeature, alpha),  # the classifier tuple
            errors)         # classifier errors, stored to prevent evaluation

        if loglevel > 0:
                logging.info('{}, {}, {}, {}'.format(
                    weighted_error, iThreshold, iDirection*iFeature, t))

    # TODO: add var names
    return bestInSample[1], stumps[bestInSample[0]], stumps[bestInSample[0]][2]


def predict(learnedClassifiers, test_data_npy='breast-cancer_test0.npy'):
    """Use the learned stumps (thresholds for features and their directions) to
    predict labels for new data.

    @ Parameters:
    learnedClassifiers: a list of 3-tuples (threshold, feature*direction,
      alpha) representing an ensemble of boosting stumps.

    @ Returns: classification error, predicted labels and errors
    """

    # TODO: use np array when given
    data = np.load(test_data_npy)

    y = data[:, 0]
    nLearnedClassifiers = len(learnedClassifiers)
    h_x = np.zeros((data.shape[0], nLearnedClassifiers))
    for iStump in range(nLearnedClassifiers):  # 0th column is the label
        iThreshold, temp, alpha = learnedClassifiers[iStump]
        iDirection = np.sign(temp)  # Extract the direction and the feature
        iFeature = np.abs(temp)

        h_i_x = (data[:, iFeature] >= iThreshold+0)*2-1  # {-1, 1} labels
        errors = (-y * h_i_x+1)/2  # {0, 1}, 1 records misclassified data
        # Invert the classifier and its errors if the classifier was inverted
        if iDirection < 0:
            errors = 1 - errors
            h_i_x = -1 * h_i_x
        h_x[:, iStump] = alpha * h_i_x            # weighted

    y_predict = np.sign(np.sum(h_x, 1))           # majority

    errors = ((y != y_predict)+0.0)*2-1
    error = np.sum((y != y_predict)+0.0)/y.shape[0]

    return error, y_predict, y, errors


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


def generateResults(g, h, data_npy, pp, gammaHistoryFile, logFilename):
    """Create result plots and tables using the ensemble g (includes alpha) and h
    (includes errors).
    Created tables:
    Created plots:
      1. Histogram of edges of classifiers in the ensemble
      2. Test and training error of the ensemble as a function of iterations t for
         <data_npy>_train0.npy and <data_npy>_test0.npy.
      3. Distribution of errors for all evaluated base classifiers at each
         round. This needs the logs to be enabled, and is computationally
         expensive to plot and render.

    """
    output = predict(g, data_npy + '_train0.npy')
    h.append((output[0], (-999, -999, -999)))
    ensembleFig = writeStumps2CSV(h, data_npy + '_ensemble')
    pp.savefig(ensembleFig)

    ng = len(g)
    error_history = np.zeros((ng, 3))
    # TODO: with linear algebra, this loop can only be evaluated once!
    for i in tqdm(range(1, ng+1)):
        # TODO: use input data rather than read
        error_train, y_predict, y, errors = predict(
            g[0:i], data_npy + '_train0.npy')
        error_test,  y_predict, y, errors = predict(
            g[0:i], data_npy + '_test0.npy')
        error_history[i-1, :] = np.array([i, error_train, error_test])

    print('The test error for {} was: {}'.format(
        data_npy+'_test0.npy', error_test))
    np.savetxt(gammaHistoryFile, error_history,
               delimiter=',', comments='',
               header='iteration, train-error, test-error')

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
            missingFeatures.nonzero()[1].size))
        data = data[:, ~missingFeatures]
    if np.isnan(data).any():
        print('There are still {} missing data points.'.format(
            np.isnan(data).nonzero()[0].size))

    return data


if __name__ == '__main__':
    helpCLI = """ Use command `python2 adaboost.py <data> 0.3 1e4 0 [--log=INFO]` to run
adaBoost with threshold functions as base classifiers on the dataset in
<data>_train0.npy then test <data>_test0.npy. At each iteration of the 1e4
total iterations, only 30% of the classifiers are evaluated randomly selected
with seed 0. You can also only provide the <data> to keep the rest as default,
or everything except the seed. Use --log=INFO to enable logging classifiers at
each iteration to inspect gamma.

Examples:
python2 adaboost.py breast-cancer 0.25
python2 adaboost.py breast-cancer 0.25 1e4
python2 adaboost.py cod-rna 0.3 1e4 1234
python2 adaboost.py cod-rna 0.3 1e4 1234 --log=INFO
"""
    print(helpCLI)
    print('-----------------------------------\n')

    nargv = len(sys.argv)
    if nargv < 6:
            loglevel = 0
    else:
            loglevel = getattr(logging, sys.argv[5][6:].upper())
            print('Logs enabled.')

    if nargv < 5:
            seed = 0
    else:
            seed = int(sys.argv[4])
    if nargv < 4:
            T = int(1e4)
    else:
            T = int(float(sys.argv[3]))
    if nargv < 3:
            sampleClassifierRatio = 0.3
    else:
            sampleClassifierRatio = float(sys.argv[2])
    if nargv < 2:
            data_npy = 'breast-cancer'
    else:
            data_npy = sys.argv[1]

    print('Training {} with {}% stumps for {} iterations ...'.format(
        data_npy, sampleClassifierRatio*100, T))
    g = adaBoost(data_npy, (0, sampleClassifierRatio), T, seed, loglevel)
