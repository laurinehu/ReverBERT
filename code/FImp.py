# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 14:16:23 2020

@author: abazin
"""

import copy
import numpy as np
from sklearn.model_selection import RandomizedSearchCV

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
def mae(V1, V2):
    S = 0
    for i in range(len(V1)):
        S += abs(V1[i]-V2[i])
    return sum([abs(V1[i]-V2[i]) for i in range(len(V1))])/len(V1)


def mse(V1, V2):
    return sum([(V1[i]-V2[i])**2 for i in range(len(V1))])/len(V1)


def rmse(V1, V2):
    return np.sqrt(mse(V1, V2))


def rsq(V1, V2):
    mean = sum(V1)/len(V1)
    SStot = sum([(v-mean)**2 for v in V1])
    SSres = sum([(V1[i]-V2[i])**2 for i in range(len(V1))])
    return 1-(SSres/SStot)

def rsqq(x_values, y_values):
    correlation_matrix = np.corrcoef(x_values, y_values)
    correlation_xy = correlation_matrix[0, 1]
    r_squared = correlation_xy**2
    return r_squared


def AvCorrCoef(labels,preds):
    R = 0
    
    labels = np.array(labels)
    preds = np.array(preds)
    
    for i in range(labels.shape[1]):
        
        avLabels = np.mean(labels[:,i])
        avPreds = np.mean(preds[:,i])
        
        num = 0        
        denum1 = 0
        denum2 = 0
        for j in range(labels.shape[0]):
            num += (labels[j,i]-avLabels)*(preds[j,i]-avPreds)
            denum1 += (labels[j,i]-avLabels)**2
            denum2 += (preds[j,i]-avPreds)**2
            
            denum = np.sqrt(denum1*denum2)

        if denum != 0:
            R += num/denum
        else:
            R = 0

    return R/labels.shape[1]




def AvRelErr(labels,preds):
    R = 0
    
    labels = np.array(labels)
    preds = np.array(preds)
    
    for i in range(labels.shape[1]):
        
        S = 0
        for j in range(labels.shape[0]):
            S += abs(labels[j,i]-preds[j,i])/labels[j,i]
        R += S/labels.shape[0]
    
    return R/labels.shape[1]




        
def MeanSqErr(labels,preds):
    R = 0
    
    labels = np.array(labels)
    preds = np.array(preds)
    
    for i in range(labels.shape[1]):
        S = 0
        for j in range(labels.shape[0]):
            S += (labels[j,i]-preds[j,i])**2
        R += S/labels.shape[0]
    
    return R



def AvRootMeanSqErr(labels,preds):
    R = 0
    
    labels = np.array(labels)
    preds = np.array(preds)
    
    for i in range(labels.shape[1]):
        S = 0
        for j in range(labels.shape[0]):
            S += (labels[j,i]-preds[j,i])**2
        R += np.sqrt(S/labels.shape[0])
    
    return R/labels.shape[1]
    
    
def AvRelRootMeanSqErr(labels,preds):
    R = 0
    
    labels = np.array(labels)
    preds = np.array(preds)
    
    for i in range(labels.shape[1]):
        S = 0
        S2 = 0
        AvLabels = np.mean(labels[:,i])
        for j in range(labels.shape[0]):
            S += (labels[j,i]-preds[j,i])**2
            S2 += (labels[j,i]-AvLabels)**2
        R += np.sqrt(S/S2)
    
    return R/labels.shape[1]  


def feature_importance(data_test, target_test, classifier, n_perm):
    R = [0]*data_test.shape[1]
    target_pred = classifier.predict(data_test)
    accu0 = accuracy(confusion_matrix(target_test, target_pred))
    for k in range(len(data_test.shape[1])):
        S = 0
        accu = 0
        data_testS = copy.deepcopy(data_test)
        for i in range(n_perm):
            data_testS = shuffle(k, data_test)
            target_pred = classifier.predict(data_testS)
            accu = accuracy(confusion_matrix(target_test, target_pred))
            S = S + (accu-accu0)
        R[k] = S
    return R

def optiRF(data,labels):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    
    criterion=['mse', 'mae']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]# Create the random grid
    random_grid = {'n_estimators': n_estimators, 'max_features': max_features, 'criterion': criterion, 'max_depth': max_depth, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf, 'bootstrap': bootstrap}
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestRegressor()
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    
    rf_random.fit(data, labels)
    
    return rf
