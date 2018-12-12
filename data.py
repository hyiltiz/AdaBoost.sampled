#!/usr/bin/env python2
import numpy as np
import sys

# use this to debug
# import pdb; pdb.set_trace()

def read(file):
	"""
	Takes a file with data in the 'libsvm' format and returns the data in a 2 dimensional ndarray.
	The first element each line is the class of data point.

	TODO: breast-cancer classifies using [2,4] instead of [-1,1] for some reason. This needs to be
	changed eventually

	@ Params: file
	@ Return: datarray
	"""
	with open(file) as datafile:
		datalines = datafile.readlines()
		datalines = [x.strip() for x in datalines]
	n_features = len(datalines[0].split())
	datarray, datalines = np.ndarray((len(datalines),n_features)), __checkdata(file)
	for i in range(len(datalines)):
		dataline_list = datalines[i].split()
		for j in range(n_features-1):
			if j >= 9:
				dataline_list[j+1] = dataline_list[j+1][3:]
			else:
				dataline_list[j+1] = dataline_list[j+1][2:]
		datarray[i,:] = dataline_list
	# normalize the data into [-1,1]
	scale = lambda x: (x-np.min(x))/(np.max(x)-np.min(x))*2-1
	return scale(datarray)

def split(file, seed=0, p_train=0.8):
	"""
	Splits a given data file into a training set and a testing set. Using 80% of the data as the training set
	and 20% of the data as the testing set. A seed can be passed as a parameter to change how the function
	random splits the data. p_train is the percentage of data points that will be training data.

	Writes the training and testing data to new files called 'file'+ '_test' + str(seed) and 'file' + '_train' + str(seed).

	@ Param: file, seed, p_train
	@ Return: data_train, data_test
	"""
	np.random.seed(seed)
	datarray, data_train, data_test = read(file), [], []
	for i in range(len(datarray)):
		select = np.random.binomial(1, p_train)
		if select == 0:
			data_test.append(datarray[i])
		else:
			data_train.append(datarray[i])
	np.save((file[:-4]+ "_train" + str(seed)+".npy"), data_train)
	np.save((file[:-4]+ "_test" + str(seed) + ".npy"), data_test)
	return data_train, data_test

def __checkdata(file):
	"""
	Takes a file with data in the 'libsvm' format and returns a list where each element is a line of the data
	This function removes all lines in the data where data for any features is missing. This function relies on the
	first line in the data having the full number of features so that needs to be checked manually beforehand.

	@ Params: file
	@ Return: datalines (missing data is removed)
	"""
	with open(file) as datafile:
		datalines = datafile.readlines()
		datalines = [x.strip() for x in datalines]
	n_features = len(datalines[0].split())
	for i,line in enumerate(datalines):
		if len(line.split()) < n_features:
			del datalines[i]
	return datalines

if __name__ == '__main__':
	if sys.argv[2] == 'r':
		read(sys.argv[1])
	elif sys.argv[2] == 'split':
		# TODO: implement split
		split(sys.argv[1])
	elif sys.arvg[2] == 'w':
		# Probably don't need to do this
		write(sys.argv[1])
	else:
		print "Use command 'python <file-name> <r,split,w>'" + \
			  "\n" + "'r'to read the data from the file and return an array" +\
			  "\n" + "'split' to split the data into a training and test data" +\
			  "\n" + "'w' to write to a data file"
