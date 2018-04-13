import os
from os import listdir, mkdir
from os.path import isfile, join
from shutil import copyfile
import numpy as np

# define path from current
current = os.getcwd()
data_dir = os.path.dirname(''.join([current,'/../notMNIST_small/']))
# or define path from absolute
#data_dir = os.path.dirname('/Users/drewthayer/galvanize/classes/dsi-cnn/data/')

# define existing relative paths
letters = ['A','B','C','D','E','F','G','H','I','J']
letterpaths = []
for letter in letters:
    letterpaths.append(data_dir + '/' + letter + '/')

# create new train/test directories
trainpath = data_dir + '/train/' # must match dirs in loop below
testpath = data_dir + '/test/'
# os.mkdir(trainpath)
# os.mkdir(testpath)
#
# for letter in letters:
#     os.mkdir(trainpath + letter + '/');
#     os.mkdir(testpath + letter + '/');
#
# print ("Test/Train paths are created")




# split data into train and test sets, write files
percent_test = .20 # of files in test
for idx, letter in enumerate(letters):
    # paths
    readpath = letterpaths[idx]
    writepath_train = trainpath + '/' + letter
    writepath_test = testpath + '/' + letter

    # read in data
    files = np.array([f for f in listdir(readpath) if isfile(join(readpath, f))])

    # do the split
    test_files = np.random.choice(files, int(percent_test*len(files)))
    for file in test_files:
        copyfile(readpath + file, writepath_test + '/' + file)

    train_files = files[~np.in1d(files, test_files)]
    for file in train_files:
        copyfile(readpath + file, writepath_train + '/' + file)
