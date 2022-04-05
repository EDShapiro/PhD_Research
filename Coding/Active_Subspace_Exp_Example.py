# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Active subspace example with the exponential function:
# f(x) = exp(c^Tx)
# c:= coefficients vector of length m
# x:= input vector of length m
# rho := sampling density of x [Uniform Densitty on the Unit-Hypercube]
# C:= Correlation matrix of dimension nxn
# alpha = oversampling factor]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def exp_f(c,x):
    t = np.dot(x,c)
    f = np.exp(t)
   
    return f,t

def grad_exp_f(c,x):
    a =  np.zeros([x.shape[0],x.shape[1]])
    #C = np.zeros([2,2])
    for i in range(0,len(x)):
        a[i,:] = [c[0]*np.exp(np.dot(x[i,:],c)), c[1]*np.exp(np.dot(x[i,:],c))]#Construct a gradient vector
        #C += np.outer(a[i,:],a[i,:]) #Construct covariance matrix
    #C = C/len(x)
    return a

def AS_Cov(x):
    #Constructing the covariance matrix based on the gradient data
    #Returns C, and the eigenvector/eigenvalue decomposition: v, W
    C = np.zeros([x.shape[1],x.shape[1]])
    for i in range(0,len(x)):
        #Construct covariance matrix
        C += np.outer(x[i,:],x[i,:])
    C = C/x.shape[0]
    v,w = np.linalg.eig(C)
    return C
   
a = np.random.uniform(-1,1,[32,2])

c = np.array([0.3,0.7])

grad = grad_exp_f(c,a)
C = AS_Cov(grad)

v,w = np.linalg.eig(C)

#f evaluated along the active direction    


f_1,t_1 = exp_f(c, a)
f_2,t_2 = exp_f(w[:,1], a)



fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(t_1, f_1, s=10, c='b', marker="o", label='Model Data')
ax1.scatter(t_2, f_2, s=10, c='r', marker="o", label='Active Subspace Data')
plt.legend(loc='upper left');
plt.show()




#d = {'col1':g_1, 'col2':g_2}
#df = pd.DataFrame(data = d)

