#!/usr/bin/env python2
import numpy as np
import sys

# use this to debug
# import pdb; pdb.set_trace()

def read(file):
    """
    Takes a file with data in the 'libsvm' format and returns the data in a 2 dimensional ndarray.
    The first element each line is the class of data point.

    @ Params: file
    @ Return: datarray
    """
    if file == 'ionosphere.txt':
        digit_start = 8
    else: 
        digit_start = 9

    with open(file) as datafile:
        datalines = datafile.readlines()
        datalines = [x.strip() for x in datalines]
    datalines = __checkdata(file)
    n_features = len(datalines[0].split())
    datarray = np.ndarray((len(datalines),n_features))
    for i in range(len(datalines)):
        dataline_list = datalines[i].split()
        for j in range(n_features-1):
            if j >= digit_start:
                dataline_list[j+1] = dataline_list[j+1][3:]
            else:
                dataline_list[j+1] = dataline_list[j+1][2:]

        datarray[i,:] = dataline_list
    # normalize the data into [-1,1]
    scale = lambda x: (x-np.min(x,0))/(np.max(x,0)-np.min(x,0))*2-1

#   import pdb; pdb.set_trace()
    # Do not scale class labels
#   datarray = np.column_stack([datarray[:,0], scale(datarray[:,1:])])

    # scale class labels as well
    # breast-cancer has labels of 2, 4

    datarray = scale(datarray)
    return datarray

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

def split_addNoise(file, seed = 0, p_train = 0.8, p_n = 20):
    np.random.seed(seed)
    p_noise = p_n/100.
    datarray, data_train, data_test = read(file), [], []
    num_switched = 0
    for i in range(len(datarray)):
        select = np.random.binomial(1, p_train)
        if select == 0:
            data_test.append(datarray[i])
        else:
            reverse = np.random.binomial(1,p_noise)
            if reverse == 1:
                if datarray[i][0] == 1:
                    datarray[i][0] = -1
                else:
                    datarray[i][0] = 1
                num_switched = num_switched + 1
            data_train.append(datarray[i])
    #print data_train
    np.save((file[:-4]+ str(p_n) + "_train" + str(seed) + ".npy"), data_train)
    np.save((file[:-4]+ str(p_n) + "_test" + str(seed) + ".npy"), data_test)
    return data_train, data_test

def multi_noise_split(file, seed = 2, p_train = 0.8, n_sigma = 50):
    np.random.seed(seed)
    datarray, data_train, data_test = read(file), [], []
    sig_array = np.linspace(0,1, n_sigma, endpoint = False)
    print sig_array
    for k in range(len(datarray)):
        select = np.random.binomial(1,p_train)
        if select == 0:
            data_test.append(datarray[k])
        else:
            data_train.append(datarray[k])
    for i in range(n_sigma):
        noisy_data = np.asarray(data_train)[:,1:] + np.random.normal(0,sig_array[i], np.asarray(data_train)[:,1:].shape)
        scale = lambda x: (x-np.min(x,0))/(np.max(x,0)-np.min(x,0))*2-1
        noisy_data = np.concatenate((np.expand_dims(np.asarray(data_train)[:,0],1),scale(noisy_data)),1)
        np.save((file[:-4] + str(i) + "_train" + str(seed) + ".npy"), noisy_data)
        np.save((file[:-4] + str(i) + "_test" + str(seed) + ".npy"), data_test)
            
def __checkdata(file):
    """
    Takes a file with data in the 'libsvm' format and returns a list where each element is a line of the data
    This function removes all lines in the data where data for any features is missing. This function relies on the
    first line in the data having the full number of features so that needs to be checked manually beforehand.

    @ Params: file
    @ Return: datalines (missing data is removed)
    """
    datalist = []
    with open(file) as datafile:
        datalines = datafile.readlines()
        datalines = [x.strip() for x in datalines]
    n_features = len(datalines[0].split())
    line_count = 0
    for i,line in enumerate(datalines):
        if len(line.split()) == n_features:
            datalist.append(line)
            line_count = line_count - 1
    return datalist

if __name__ == '__main__':
    if sys.argv[2] == 'r':
        read(sys.argv[1])
    elif sys.argv[2] == 'split':
        # TODO: implement split
        split(sys.argv[1])
    elif sys.argv[2] == 'w':
        # Probably don't need to do this
        write(sys.argv[1])
    elif sys.argv[2] == 'splitn':
        split_addNoise(sys.argv[1])
    elif sys.argv[2] == 'norm-noise':
        multi_noise_split(sys.argv[1])
    else:
        print "Use command 'python <file-name> <r,split,w>'" + \
              "\n" + "'r'to read the data from the file and return an array" +\
              "\n" + "'split' to split the data into a training and test data" +\
              "\n" + "'splitn' to split the data and add noise" \
              "\n"
        # no_noise = np.load('breast-cancer0_train0.npy')
        # lotso_noise = np.load('breast-cancer49_train0.npy')
        # print no_noise
        # print lotso_noise