import numpy as np  
import sys

def read(file):
	"""
	Takes a file with data in the 'libsvm' format and returns the data in a 2 dimensional ndarray
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
			dataline_list[j+1] = dataline_list[j+1][2:]
		datarray[i,:] = dataline_list
	return datarray

def __checkdata(file):
	"""
	Takes a file with data in the 'libsvm' format and returns a list where each element is a line of the data
	This function removes all lines in the data where data for any features is missing
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