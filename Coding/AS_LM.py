# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 00:21:44 2022

@author: 17204
"""
#Active Subspaces Using Local Linear Models
# alpha:= Sampling factor 1 (should be between (2,10) = 10
# beta := Over Sampling factor 2 (1,\inty)
# m:= Dimension of input space
# k:= Maximum dimension +1 we wish to work with after applying AS
# N:= Local regression samples = beta*alpha*m 
# M:= Fixed point samples:=alpha*k*log(m)
# p:= Number of points to form local linear models with
# q:= quantity of interest from model a given model

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

def exp_f_2(c,x):
    X_trans = np.zeros_like(x)
    X_trans[:,0] = x[:,0]**2
    X_trans[:,1] = x[:,1]**2
    X_trans[:,2] = x[:,2]
    x_T_c = np.dot(X_trans,c) + np.random.normal(0,1, X_temp.shape[0])
    #plt.scatter(X_trans[:,1], x_T_c)
    #f = np.exp(x_T_c) 
    return x_T_c

def lin_coef(X,y):
    X_N = np.zeros([X.shape[0], X.shape[1]+1])
    X_N[:,1:4] = X
    X_N[:,0] = np.ones(X.shape[0])
    X_T_X = np.dot(X_N.T,X_N)
    theta = np.dot(np.linalg.inv(X_T_X), np.dot(X_N.T,y))
    return theta

alpha = 10
beta = 10
k = 3
m = 3
N = beta*alpha*m
M = alpha*k*np.log(m)
p = m+1


mu = np.array([0,0.1,0.3])
#Covariance arra
cov_m_r = np.array([[1,0.94,0.01],[0.94,1,0], [0.01,0,1]])

X_M = np.random.multivariate_normal(mu, cov_m_r, int(np.ceil(M)))
X_N = np.random.multivariate_normal(mu, cov_m_r,N)

c = (0.8, 0.3, 0.01)

#Create an active subspace based on local linear models of data points
#Finding all distance between points in X_M and X_N
#for j in range X_M.shape[0]:


local = np.zeros(X_N.shape[0])

for j in range(X_M.shape[0]):
    for i in range(X_N.shape[0]):
    #print(np.linalg.norm(X_N[i,:] - X_M[1]))
        local[i] = np.linalg.norm(X_N[i,:] - X_M[j,:]) 
        
    b = np.zeros([X_M.shape[1], X_M.shape[1]])
    X_temp = np.zeros([p,X_M.shape[1]])    
    
    
    local_s = local.copy()
    local_s.sort()
    #Finding indices that correspond to the 20 points in X_N that are
    # closest to the point X_M[i] from the chosen point
    wtvr, loc_s_i, loc_og_i = np.intersect1d(local_s,local, return_indices = True)
    for i in range(0,p,1):
        X_temp[i,:] = X_N[loc_og_i[i],:]
    

   
    #y_temp = reg_2.predict(X_temp)
    y_temp = exp_f_2(c,X_temp)
    theta = lin_coef(X_temp, y_temp)
    #Create local linear model based on temporary data
    
    print(theta)
    b += np.outer(theta[1:4],theta[1:4])
C = b/X_M.shape[0]

v,w = np.linalg.eig(C)
print(w)




    