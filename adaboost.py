import numpy as np
import matplotlib.pyplot as plt
import sys
import datetime
import logging

# use this to debug
# import pdb; pdb.set_trace()

def adaBoost(data_npy='breast-cancer_train0.npy', sampleRatio=(0,0), T=int(1e2), seed=0, loglevel=0):
    """Takes an input data file storing numpy array and a function that generates
    base classifiers, evaluates all base classifiers when isSample is False,
    and runs classic AdaBoost.

    @ Return: learned classifier g that is a function that takes features and
    returns a prediction

    """
    # import pdb; pdb.set_trace()
    data = np.load(data_npy)
    m_samples = data.shape[0]
    logging.basicConfig(format='%(message)s',
                        filename='{}classifiers_history_seed{}_sampleRatio{}_{}.log'.format(data_npy, seed, sampleRatio[1], datetime.datetime.now().isoformat()),
                        level=loglevel)
    logging.info('weighted_error, threshold, feature_direction, iteration')

    stumps = createBoostingStumps(data)
    print('Number of base classifiers: {}'.format(len(stumps)))
    writeStumps2CSV(stumps, data_npy)

    # AdaBoost algorithm
    D_t = np.zeros(m_samples)+1.0/m_samples
    h=[]
    a=np.zeros(T)
    for t in range(0,T):
        e_t, h_t, errors = evalToPickClassifier(stumps, D_t, data, sampleRatio, seed, t)
        h.append(h_t[0:2]) # keep the errors
#       import pdb; pdb.set_trace()
        a[t] = 1.0/2 * np.log(1/e_t-1)
        # not keeping track of D_t,Z_t history to optimize for memory
        Z_t = 2*np.sqrt(e_t*(1-e_t))
        D_t = D_t * np.exp(a[t]*(2*errors-1))/Z_t # errors := (-y * h_i_x+1)/2

    g = [(hi[1][0], hi[1][1], a[i]) for i, hi in enumerate(h)]
    output = predict(g, data_npy)
    h.append((output[0], (-999, -999, -999)))
    # import pdb; pdb.set_trace()
    writeStumps2CSV(h, data_npy + '_ensemble')
    return g

def createBoostingStumps(data):
    """
    Create boosting stumps, i.e. axis aligned thresholds for each features.
    Sorts the data first and uses the sorted values for each component to
    construct the thresholds. Has complexity O(mNlogm).
    """
    import pdb; pdb.set_trace()
    baseClassifiers = []
    y = data[:,0]
    D_t = 1.0/data.shape[0]
    # NOTE: these loops can run in parallel
    for iFeature in range(1,data.shape[1]): # 0th column is the label
        thresholds = np.sort(data[:,iFeature])
        for iThreshold in thresholds:
            iDirection = +1
            h_i_x = ((thresholds >= iThreshold)+0)*2-1
            errors = (-y * h_i_x+1)/2
            weighted_error = sum(D_t * errors)
            if weighted_error > 0.5:
                iDirection = -iDirection # invert the classifier
                errors = 1-errors
                weighted_error = sum(D_t * errors)
            weight = 1.0 # stores alpha, not used until predict() i.e. until adaBoost() finishes training
            # weighted_error weights classification errors by D_t
            # Here D_t is uniform to create the stumps
            baseClassifiers.append((weighted_error, (iThreshold, iDirection*iFeature,weight), errors))
            logging.info('{}, {}, {}, {}'.format(weighted_error, iThreshold, iDirection*iFeature, 0))

    return baseClassifiers

def evalToPickClassifier(stumps, D_t, data, sampleRatio, seed, t):
    """
    This function currently samples the data. Could also sample the classifiers.
    """
    np.random.seed(seed)
    sampleDataRatio, sampleClassifierRatio = sampleRatio

    # NOTE: sampling data is not implemented yet
    if sampleDataRatio == 0:
        # treat these the same: no sampling or sample everything
        sampleDataRatio == 1
    index_data = np.random.rand(1, data.shape[1]-1) < sampleDataRatio

    if sampleClassifierRatio == 0:
    # treat these the same: no sampling or sample everything
        sampleClassifierRatio = 1
#   import pdb; pdb.set_trace()
    index_classifiers = np.random.rand(1, len(stumps)) < sampleClassifierRatio
    index_classifiers_list = np.ndarray.tolist(np.where(index_classifiers)[1])

    y = data[:,0]
    # D_t = 1.0/data.shape[0]

    # evaluate a subset of classifiers
    # keeping track of the smallest error
    bestInSample = (-1,1)
    # NOTE: these loops can run in parallel
    for iStump in index_classifiers_list: # 0th column is the label
        # load a classifier
        iThreshold,temp,weight = stumps[iStump][1]
        iFeature = np.abs(temp)
        iDirection = np.sign(temp)
        errors = stumps[iStump][2]

        weighted_error = sum(D_t * errors)
        if weighted_error > 0.5:
            iDirection = -iDirection # invert the classifier
            errors = 1 - errors
            weighted_error = sum(D_t * errors)

        if weighted_error < bestInSample[1]:
            bestInSample = (iStump, weighted_error)

        # update this classifier
        weight = 1.0
        stumps[iStump] = (weighted_error, (iThreshold, iDirection*iFeature,weight), errors)
        logging.info('{}, {}, {}, {}'.format(weighted_error, iThreshold, iDirection*iFeature, t))

    return bestInSample[1], stumps[bestInSample[0]], stumps[bestInSample[0]][2]

def predict(learnedClassifiers, test_data_npy='breast-cancer_test0.npy'):
    """
    Use the learned stumps (thresholds for features and their directions) to
    predict labels for new data.
    """
    # import pdb; pdb.set_trace()
    data = np.load(test_data_npy)
    y = data[:,0]
    h_x = np.zeros((data.shape[0], len(learnedClassifiers)))
    for iStump in range(len(learnedClassifiers)): # 0th column is the label
        iThreshold,temp,weight = learnedClassifiers[iStump]
        iDirection = np.sign(temp)
        iFeature = np.abs(temp)

        h_i_x =( data[:,iFeature] >= iThreshold+0)*2-1
        errors = (-y * h_i_x+1)/2
        if iDirection < 0:
            errors = 1 - errors
        h_x[:,iStump] = weight*((errors+0)*2-1)

    # import pdb; pdb.set_trace()
    y_predict = np.sign(np.sum(h_x, 1))
    errors = (y != y_predict+0)*2-1
    error = np.sum(y != y_predict)/y.shape[0]
    return error, y_predict, y, errors

def writeStumps2CSV(stumps, data_npy):
    # stumpsTable = np.zeros((len(stumps), 5))
    stumpsTable = np.zeros((len(stumps), 4))
    for iStump in range(len(stumps)):
            stumpsTable[iStump,:] = np.hstack((
                    np.array(stumps[iStump][0]), # weighted_error
                    np.array(stumps[iStump][1])  # stumps (threshold, feature*direction, alpha)
                    # np.sum(stumps[iStump][2])/stumps[iStump][2].shape[0] # raw error; not needed
            ))

    # 0.5-weighted_error=gamma when D_t is uniform, i.e. for base classifiers
    # this does not hold for g, i.e. post training classifiers
    np.savetxt(data_npy + '_stumps.log.csv', stumpsTable, delimiter=',', newline='\n', comments='',
                  # header='weighted_error, threshold, feature, weight_alpha, error')
                  header='weighted_error, threshold, feature, weight_alpha')
    plt.hist(0.5 - stumpsTable[:,0])
    plt.ylabel('gamma')


if __name__ == '__main__':
    # import pdb; pdb.set_trace()
    print "Use command 'python2 adaboost.py <data> 0.3 1e4 0 [--log=INFO]'" + \
        "\n" + "to run adaBoost with threshold functions as base classifiers on" + \
        "\n" + "the dataset in <data>_train0.npy then test <data>_test0.npy." + \
        "\n" + "At each iteration of the 1e4 total iterations," + \
        "\n" + "only 30% of the classifiers are evaluated randomly selected with seed 0." + \
        "\n" + "You can also only provide the <data> to keep the rest as default, or everything except the seed." + \
        "\n" + "Use --log=INFO to enable logging classifiers at each iteration to inspect gamma." + \
        "\n" + "Examples:" + \
        "\n" + "python2 adaboost.py" + \
        "\n" + "python2 adaboost.py breast-cancer 0.25" + \
        "\n" + "python2 adaboost.py breast-cancer 0.25 1e4" + \
        "\n" + "python2 adaboost.py cod-rna 0.3 1e4 1234" + \
        "\n" + "python2 adaboost.py cod-rna 0.3 1e4 1234 --log=INFO"
    print('-----------------------------------\n')
    if len(sys.argv) < 6:
            loglevel = 0
    else:
            loglevel = getattr(logging, sys.argv[5][6:].upper())
            print('Logs enabled.')

    if len(sys.argv) < 5:
            seed = 0
    else:
            seed = int(sys.argv[4])
    if len(sys.argv) < 4:
            T = int(1e4)
    else:
            T = int(float(sys.argv[3]))
    if len(sys.argv) < 3:
            sampleClassifierRatio = 0.3
    else:
            sampleClassifierRatio = float(sys.argv[2])
    if len(sys.argv) < 2:
            data_npy = 'breast-cancer'
    else:
            data_npy = sys.argv[1]

    print('Training {} with {}% stumps for {} iterations ...'.format(data_npy, sampleClassifierRatio*100, T))
    g = adaBoost(data_npy + '_train0.npy', (0, sampleClassifierRatio), T, seed, loglevel)
    error, y_predict, y, errors= predict(g, data_npy + '_test0.npy')
    print('The error for {} was: {}'.format(data_npy+'_test0.npy', error))

    # to plot the history
    # x = np.loadtxt('breast-cancer_train0.npyclassifiers_history_seed0_sampleRatio1.0_2018-12-14T20:42:40.856094.log', delimiter = ',', skiprows = 1)
    # plt.scatter(x[:,3], x[:,0], s=0.1),plt.xlabel('Iterations'),plt.ylabel('Errors'),plt.title('Training history for breast-cancer dataset')
