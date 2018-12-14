import numpy as np
import sys
import warnings
from hypothesis import generate_baseclassifiers

# use this to debug
# import pdb; pdb.set_trace()

def adaBoost(data_npy='breast-cancer_test0.npy', sampleRatio=(0,0), T=int(1e4), seed=0):
	"""Takes an input data file storing numpy array and a function that generates
	base classifiers, evaluates all base classifiers when isSample is False,
	and runs classic AdaBoost.

	@ Return: learned classifier g that is a function that takes features and
	returns a prediction

	"""
#	import pdb; pdb.set_trace()
	data = np.load(data_npy)
	m_samples, N_feats = data.shape

	thresholds = generate_baseclassifiers(data_npy, 1e2)
	directions = np.zeros(thresholds.shape, int)
	classifiers = (thresholds, directions)

	# AdaBoost algorithm
	D_t = np.zeros(m_samples)+1.0/m_samples
	h=[]
	a=np.zeros(T)
	for t in range(0,T):
		e_t, h_t, h_t_x = evalToPickClassifier(classifiers, D_t, data, sampleRatio, seed)
		h.append(h_t)
#		import pdb; pdb.set_trace()
		a[t] = 1.0/2 * np.log(1/e_t-1)
		# not keeping track of D_t,Z_t history to optimize for memory
		Z_t = 2*np.sqrt(e_t*(1-e_t))
		D_t = D_t * np.exp(-a[t]*data[:,0] * h_t_x)

	import pdb; pdb.set_trace()
	g = np.prod(a, h) # NOTE: psudocode
	return g


def evalToPickClassifier(classifiers, D_t, data, sampleRatio, seed):
	"""
	This function currently samples the data. Could also sample the classifiers.
	"""
#	import pdb; pdb.set_trace()
#	np.random.seed(seed)
#	sampleDataRatio, sampleClassifierRatio = sampleRatio
#	if sampleDataRatio == 0:
#		# treat these the same: no sampling or sample everything
#		sampleDataRatio == 1
#	index_data = np.random.rand(1, data.shape[1]-1) < sampleDataRatio
#
#	if sampleClassifierRatio == 0:
#	# treat these the same: no sampling or sample everything
#		sampleClassifierRatio == 1
#	index_classifiers = np.random.rand(1, classifiers.shape[1]-1) < sampleClassifierRatio # NOTE: shape is wrong
#
#	# only use the subsets
#	isCorrect = compare(data[index_data], classifiers[index_classifiers]) # NOTE: to be implemented
#	error = np.sum(not isCorrect)
#	h_t = np.argmin(error)
#	e_t = np.min(error)
#
#	return e_t, h_t
	h_t_x = np.random.binomial(1,0.6,data.shape[0])*2-1
	return (0.4, (0.1,+1, 1),h_t_x) # returns (error, (threshold,direction*feature, weight),[h(x_i)])

if __name__ == '__main__':
	#if isinstance(sys.argv[1], basestring):
	adaBoost()
	#else:
	#	print "Use command 'python <file-name> train.npy (0.1,0.3) 0'" + \
	#		  "\n" + "to run adaBoost with threshold functions as base classifiers on" +\
	#		  "\n" + "the dataset in train.npy. At each epoch, 10% of the data are used." +\
	#		  "\n" + "and 30% of the classifiers are being evaluated."
