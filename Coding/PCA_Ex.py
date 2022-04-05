#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 14:36:08 2022

@author: a17204
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import PCA.decomposition


mu = np.array([0,0.1,0.3])
#Covariance arra
cov_m_r = np.array([[1,-0.9,0],[-0.9,1,0], [0,0,1]])
corr_data = np.random.multivariate_normal(mu, cov_m_r,(1000))

#print(corr_data)


def standardize_data(X):
         
    '''
    This function standardize an array, its substracts mean value, 
    and then divide the standard deviation.
    
    param 1: array 
    return: standardized array
    '''    
    rows, columns = X.shape
    
    standardizedArray = np.zeros(shape=(rows, columns))
    tempArray = np.zeros(rows)
    
    for column in range(columns):
        
        mean = np.mean(X[:,column])
        std = np.std(X[:,column])
        tempArray = np.empty(0)
        
        for element in X[:,column]:
            
            tempArray = np.append(tempArray, ((element - mean) / std))
 
        standardizedArray[:,column] = tempArray
    
    return standardizedArray

    
X = standardize_data(corr_data)
cov_m_inf = np.cov(X.T)
e_val, e_vec = np.linalg.eig(cov_m_inf)


#Choosing a subset of the principal components
W = np.zeros([3,2])
W[:,0] = e_vec[:,0]
W[:,1] = e_vec[:,1]


np.dot(X,W)

var_perc = []
for i in e_val:
    var_perc.append(i/sum(e_val)*100)
print('Variance percentage per eigenvalue=',var_perc )
#Constructing a 3x3 covariance matrix that has two principal components  