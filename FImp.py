# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 14:16:23 2020

@author: abazin
"""

import copy
import numpy as np

def shuffle(k, X):
    Y = copy.deepcopy(X)
    np.random.shuffle(Y[:, k])
    return Y


#Computes the confusion matrix from the predicted and true classes
def confusion_matrix(Y_true, Y_pred):
    M = [[0, 0], [0, 0]]
    for k in range(len(Y_true)):
        if Y_pred[k] >= 0.5:
            if Y_true[k] >= 0.5:
                M[0][0] = M[0][0] + 1
            else:
                M[0][1] = M[0][1] + 1
        else:
            if Y_true[k] >= 0.5:
                M[1][0] = M[1][0] + 1
            else:
                M[1][1] = M[1][1] + 1
    return M


#Computes the sensitivity of the classifier from the confusion matrix
def accuracy(confusion_matrix):
    TP = confusion_matrix[0][0]
    FP = confusion_matrix[0][1]
    FN = confusion_matrix[1][0]
    TN = confusion_matrix[1][1]
    Tot_pop = TP + FP + FN + TN
    try:
        R = (TP + TN)/Tot_pop
    except:
        R = 0
    return R


#V1 is ground truth, V2 is predicted
def mae(V1,V2):
    S = 0
    for i in range(len(V1)):
        S += abs(V1[i]-V2[i])
    return sum([abs(V1[i]-V2[i]) for i in range(len(V1))])/len(V1)


def mse(V1,V2):
    return sum([(V1[i]-V2[i])**2 for i in range(len(V1))])/len(V1)


def rmse(V1,V2):
    return np.sqrt(mse(V1,V2))


def rsq(V1,V2):
    mean = sum(V1)/len(V1)
    SStot = sum([(v-mean)**2 for v in V1])
    SSres = sum([(V1[i]-V2[i])**2 for i in range(len(V1))])
    return 1-(SSres/SStot)
    


def feature_importance(data_test, target_test, classifier, n_perm):
    
    R = [0]*data_test.shape[1] 
    target_pred = classifier(data_test)   
    accu0 = accuracy(confusion_matrix(target_test, target_pred)) 
    for k in range(len(data_test.shape[1])):  
        S = 0       
        accu = 0       
        data_testS = copy.deepcopy(data_test)
        for i in range(n_perm):                   
            data_testS = shuffle(k, data_test) 
            #Do something to make data_testS be in the right format
            target_pred = classifier(data_testS)           
            accu = accuracy(confusion_matrix(target_test, target_pred))           
            S = S + (accu-accu0)           
        R[k] = S       
    return R
