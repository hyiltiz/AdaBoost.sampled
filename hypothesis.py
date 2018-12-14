import numpy as np
import sys

def generate_baseclassifiers(file, n):
	"""
	Given a '.npy' file with data stored in a 2D array where each sample is a row in the data,
    this function generates a set of base classifiers of threshold functions that separate
    the range over which each features spans into 'n' possible thresholds.

	@ Param: file, n
	@ Return: 2D array of feature x threshold
	"""
	data, n = np.load(file), int(n)
	feature_ranges,	N_feats = np.zeros((2,data.shape[1]-1)),data.shape[1]-1
	for j in range(N_feats):
		for i in range(data.shape[0]):
			if data[i,j+1] < feature_ranges[0,j]:
				feature_ranges[0,j] = data[i,j+1]
			elif data[i,j+1] > feature_ranges[1,j]:
				feature_ranges[1,j] = data[i,j+1]

	classifiers = np.zeros((n,N_feats)) # first column of data is labels!

	for k in range(N_feats):
		classifiers[:,k] = np.transpose(np.linspace(feature_ranges[0,k], feature_ranges[1,k], n))

	return classifiers

if __name__ == '__main__':
	generate_baseclassifiers(sys.argv[1],sys.argv[2])
