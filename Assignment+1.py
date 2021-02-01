#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 10:47:49 2021

@author: carllos
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

def answer_zero(cancer):
    # This function returns the number of features of the breast cancer dataset, which is an integer. 
    # The assignment question description will tell you the general format the autograder is expecting
    return len(cancer['feature_names'])

def answer_one():
    # This fuction returns the sklearn.dataset cancer converted to DataFrame type.
    data = pd.DataFrame(data = cancer.data, columns = [cancer.feature_names])
    
    data['target'] = pd.Series(cancer.target)
    
    
    return data

def answer_two():
    #This function return a Series with the class distribution
    cancerdf = answer_one()
    
    counts = cancerdf.target.value_counts(ascending=True)
    counts.index = "malignant benign".split()
    return counts

def answer_three():
    #This function split the dataframe into X and y, features and labels.
    cancerdf = answer_one()
    
    X = cancerdf.iloc[:,:30]
    y = cancerdf['target']
    
    return X,y

from sklearn.model_selection import train_test_split

def answer_four():
    #This function do the train_test_split
    X, y = answer_three()
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 0)
    
    return X_train, X_test, y_train, y_test

def answer_five():
    #this function run the knn classifier and fit the train dataframes
    X_train, X_test, y_train, y_test = answer_four()
    
    knn = KNeighborsClassifier(n_neighbors = 1)
    knn.fit(X_train,y_train)
    
    return knn

def answer_six():
    #this function, with the use of knn classifier, predict the labels with the mean of each feature
    cancerdf = answer_one()
    knn = answer_five()
    means = cancerdf.mean()[:-1].values.reshape(1, -1)
    
    prediction = knn.predict(means)
    
    return prediction

def answer_seven():
    #this function returns the predict of X_test
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    
    
    return knn.predict(X_test)

def answer_eight():
    #function returns the score of the prediction
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    
    
    return knn.score(X_test,y_test)
