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
	import pdb; pdb.set_trace()
	data = np.load(data_npy)
	m_samples = data.shape[0]

	stumps = createBoostingStumps(data)


	# AdaBoost algorithm
	D_t = np.zeros(m_samples)+1.0/m_samples
	h=[]
	a=np.zeros(T)
	for t in range(0,T):
		e_t, h_t, h_t_x = evalToPickClassifier(stumps, D_t, data, sampleRatio, seed)
		h.append(h_t)
#		import pdb; pdb.set_trace()
		a[t] = 1.0/2 * np.log(1/e_t-1)
		# not keeping track of D_t,Z_t history to optimize for memory
		Z_t = 2*np.sqrt(e_t*(1-e_t))
		D_t = D_t * np.exp(-a[t]*data[:,0] * h_t_x)/Z_t

	import pdb; pdb.set_trace()
	g = np.prod(a, h) # NOTE: psudocode
	return g

def createBoostingStumps(data):
	"""
	Create boosting stumps, i.e. axis aligned thresholds for each features.
	Sorts the data first and uses the sorted values for each component to
	construct the thresholds. Has complexity O(mNlogm).
	"""
#	import pdb; pdb.set_trace()
	baseClassifiers = []
	y = data[:,0]
	D_t = 1.0/data.shape[0]
	for iFeature in range(1,data.shape[1]): # 0th column is the label
		thresholds = np.sort(data[:,iFeature])
		for iThreshold in thresholds:
			iDirection = +1
			h_i_x = ((thresholds >= iThreshold)+0)*2-1
			errors = (-y * h_i_x+1)/2
			weighted_error = sum(D_t * (-y * h_i_x+1)/2)
			if error > 0.5:
				iDirection = -1 # invert the classifier
			weight = 1.0
			baseClassifiers.append((weighted_error, (iThreshold, iDirection*iFeature,weight), errors))

	return baseClassifiers

def evalToPickClassifier(stumps, D_t, data, sampleRatio, seed):
	"""
	This function currently samples the data. Could also sample the classifiers.
	"""
	import pdb; pdb.set_trace()
	np.random.seed(seed)
	sampleDataRatio, sampleClassifierRatio = sampleRatio
	if sampleDataRatio == 0:
		# treat these the same: no sampling or sample everything
		sampleDataRatio == 1
	index_data = np.random.rand(1, data.shape[1]-1) < sampleDataRatio

	if sampleClassifierRatio == 0:
	# treat these the same: no sampling or sample everything
		sampleClassifie
	index_classifiers = np.random.rand(1, len(classifiers)) < sampleClassifierRatio
	index_classifiers_list = np.ndarray.tolist(np.where(index_classifiers)[0])

	y = data[:,0]
	# D_t = 1.0/data.shape[0]

	# evaluate a subset of classifiers
	# keeping track of the smallest error
	bestInSample = (nan,1)
	for iStump in index_classifiers_list: # 0th column is the label
		# load a classifier
		iThreshold,temp,weight = stumps[iStump][1]
		iFeature = np.abs(temp)
		iDirection = temp > 0
		errors = stumps[iStump][2]

		weighted_error = sum(D_t * errors)
		if weighted_error > 0.5:
			iDirection = -iDirection # invert the classifier
			weighted_error = 1 - weighted_error

		if weighted_error < bestInSample[1]:
			bestInSample = (iStump, weighted_error)

		# update this classifier
		weight = 1.0
		stumps[iStump] = (weighted_error, (iThreshold, iDirection*iFeature,weight), errors)

	return bestInSample[1], stumps[bestInSample[0]], errors

if __name__ == '__main__':
	if len(sys.argv) >= 0:
	adaBoost()
	else:
		print "Use command 'python <file-name> train.npy (0.1,0.3) 0'" + \
			  "\n" + "to run adaBoost with threshold functions as base classifiers on" +\
			  "\n" + "the dataset in train.npy. At each epoch, 10% of the data are used." +\
			  "\n" + "and 30% of the classifiers are being evaluated."
