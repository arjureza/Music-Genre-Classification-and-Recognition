from python_speech_features import mfcc
from tempfile import TemporaryFile
import scipy.io.wavfile as wav
import numpy as np
import os
import pickle
import random 
import operator
import math
from collections import defaultdict

def distance(instance1 , instance2 , k):
    distance = 0 
    mm1 = instance1[0] 
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]
    distance = np.trace(np.dot(np.linalg.inv(cm2), cm1)) 
    distance += (np.dot(np.dot((mm2-mm1).transpose(), np.linalg.inv(cm2)), mm2-mm1)) 
    distance += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance -= k
    return distance

def knn(trainingSet, instance, k):
    distances = []
    for x in range (len(trainingSet)):
        dist = distance(trainingSet[x], instance, k) + distance(instance, trainingSet[x], k)
        distances.append((trainingSet[x][2], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def nearestClass(neighbors):
    classVote = {}
    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVote:
            classVote[response] += 1 
        else:
            classVote[response] = 1
    sorter = sorted(classVote.items(), key = operator.itemgetter(1), reverse=True)
    return sorter[0][0]

def loadDataset(filename):
    with open("GTZAN.dat" , 'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break

directory = "C:/Users/rezaa/OneDrive/Desktop/Auburn Spring 2021/Machine Learning/Final Project/genres/"
f = open("GTZAN.dat" ,'wb')
i = 0

# creates file with information from datasets
for folder in os.listdir(directory):
    i += 1
    if i == 11:
        break   
    for file in os.listdir(directory+folder):  
        (rate,sig) = wav.read(directory + folder + "/" + file)
        mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy = False)
        covariance = np.cov(np.matrix.transpose(mfcc_feat))
        mean_matrix = mfcc_feat.mean(0)
        feature = (mean_matrix, covariance, i)
        pickle.dump(feature, f)
f.close()

# loads dataset and splits it into training and testing
dataset = []
loadDataset("GTZAN.dat")

# testing with a new wav file
results=defaultdict(int)

# path to file
directory = "C:/Users/rezaa/OneDrive/Desktop/Auburn Spring 2021/Machine Learning/Final Project/genres/"

# associating number from 1-10 with genre
i = 1
for folder in os.listdir(directory):
    results[i] = folder
    i+=1

(rate,sig) = wav.read("C:/Users/rezaa/OneDrive/Desktop/Auburn Spring 2021/Machine Learning/Final Project/test.wav")
mfcc_feat = mfcc(sig,rate,winlen=0.020,appendEnergy=False)
covariance = np.cov(np.matrix.transpose(mfcc_feat))
mean_matrix = mfcc_feat.mean(0)

feature = (mean_matrix, covariance, 0)

prediction = nearestClass(knn(dataset, feature, 5))
print(results[prediction])