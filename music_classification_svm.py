from python_speech_features import mfcc
from tempfile import TemporaryFile
import scipy.io.wavfile as wav
import numpy as np
import os
import pickle
import random 
import operator
import math
from sklearn import svm, datasets
import time

def getAccuracy(testSet, predictions):
    correct = 0 
    for x in range (len(testSet)):
        if testSet[x][-1]==predictions[x]:
            correct+=1
    return 1.0 * correct/len(testSet)

def loadDataset(filename, split, X_train, y_train, X_test, y_test):
    with open("my.dat" , 'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break  
    for x in range(len(dataset)):
        sample = [dataset[x][0], dataset[x][1]]
        if random.random() < split:
            X_train.append(sample)
            y_train.append(dataset[x][2])    
        else:
            X_test.append(sample)
            y_test.append(dataset[x][2])

# measuring run time
start_time = time.time()

directory = "C:/Users/rezaa/OneDrive/Desktop/Auburn Spring 2021/Machine Learning/Final Project/genres/"
f= open("my.dat" ,'wb')
i=0
for folder in os.listdir(directory):
    i += 1
    if i == 11:
        break   
    for file in os.listdir(directory+folder):  
        (rate,sig) = wav.read(directory + folder + "/" + file)
        mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy = False)
        covariance = np.cov(np.matrix.transpose(mfcc_feat))
        mean_matrix = mfcc_feat.mean(0)
        covariance_mean = covariance.mean(0)
        feature = (mean_matrix, covariance_mean, i)
        pickle.dump(feature, f)
f.close()

dataset = []
X_train = []
y_train = []
X_test = []
y_test = []
loadDataset("my.dat" , 0.8, X_train, y_train, X_test, y_test)

X_train_np = np.asarray(X_train)
X_test_np = np.asarray(X_test)

nsamples, nx, ny = X_train_np.shape
X_train_two_dim = X_train_np.reshape((nsamples, nx*ny))

nsamples2, nx2, ny2 = X_test_np.shape
X_test_two_dim = X_test_np.reshape((nsamples2, nx2*ny2))

# SVM

poly = svm.SVC(kernel='poly', degree=3, C=1, decision_function_shape='ovo').fit(X_train_two_dim, y_train)
poly_pred = poly.predict(X_test_two_dim)

'''
1 = blues
2 = classical
3 = country
4 = disco
5 = hiphop
6 = jazz
7 = metal
8 = pop
9 = reggae
10 = rock
'''

accuracy_poly = poly.score(X_test_two_dim, y_test)
print(accuracy_poly)

# prints the runtime of the program
print("--- Run Time: %s seconds ---" % (time.time() - start_time))