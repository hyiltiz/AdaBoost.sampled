import numpy as np
import sys
import warnings
from hypothesis import generate_baseclassifiers

# use this to debug
# import pdb; pdb.set_trace()

def adaBoost(data_npy, sampleRatio=(0,0), T=1e4, seed=0):
	"""Takes an input data file storing numpy array and a function that generates
    base classifiers, evaluates all base classifiers when isSample is False,
    and runs classic AdaBoost.

	@ Return: learned classifier g that is a function that takes features and
	returns a prediction

    """
	data = np.load(file_npy)
	N_feats, m_samples = data.shape[2]-1,data.shape[1]-1
	classifiers = generate_baseclassifiers(data_npy)

    # AdaBoost algorithm
	D_t = np.zeros((1,m_samples))+1/m
    for t in range(1,T+1):
        e_t, h[t] = evalToPickClassifier(classifiers, D_t, data, sampleRatio, seed)
        a[t] = 1/2 * log(1/e_t-1)
        # not keeping track of D_t,Z_t history to optimize for memory
        Z_t = 2*np.sqrt(e_t*(1-e_t))
        D_t = D_t * np.exp(-a_t*data[1,]*h_t(data[2:end,])) # NOTE: psudocode 

    g = np.prod(a_t, h_t)
    return g

    import pdb; pdb.set_trace()

def evalToPickClassifier(classifiers, D_t, data, sampleRatio, seed):
    """
    This function currently samples the data. Could also sample the classifiers.
    """
	np.random.seed(seed)
    sampleDataRatio, sampleClassifierRatio = sampleRatio
    if sampleDataRatio == 0:
        # treat these the same: no sampling or sample everything
        sampleDataRatio == 1
    index_data = np.random.rand(1, data.shape[1]-1) < sampleDataRatio

    if sampleClassifierRatio == 0:
        # treat these the same: no sampling or sample everything
        sampleClassifierRatio == 1
    index_classifiers = np.random.rand(1, classifiers.shape[1]-1) < sampleClassifierRatio # NOTE: shape is wrong

    # only use the subsets
    isCorrect = compare(data[index_data], classifiers[index_classifiers]) # NOTE: to be implemented
    error = np.sum(not isCorrect)
    h_t = np.argmin(error)
    e_t = np.min(error)

    return e_t, h_t

if __name__ == '__main__':
	if isinstance(sys.argv[1], basestring):
		adaBoost(sys.argv[1], )
	else:
		print "Use command 'python <file-name> train.npy (0.1,0.3) 0'" + \
			  "\n" + "to run adaBoost with threshold functions as base classifiers on" +\
			  "\n" + "the dataset in train.npy. At each epoch, 10% of the data are used." +\
			  "\n" + "and 30% of the classifiers are being evaluated."
