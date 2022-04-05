#Linear Regression on Transformed Data Following Principal Component Analysis
# Generate random 3-D data with 2 Principal Components
# Map data through exponential function
#Find singular value decomposition of input data space
#Project daata into tranformed data space
#Peform regression between transformed data and map of original data.
#This regression is the black box function f used to construct the active
#subspace using algorithm 1.2.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

#Projection function
def proj(A,X):
    proj = np.dot(A,X.T)
    return proj

#Function for discovering active subspace
def exp_f(c,x):
    X_trans = np.zeros_like(x)
    
    x_T_c = np.dot(x,c) + np.random.normal(0,1,1000)
    f = np.exp(x_T_c) 
    return x_T_c, f

#Creates the function y = exp(ax_1 + bx_2^2 + cx_3^3 + \eps )
def exp_f_2(c,x):
    X_trans = np.zeros_like(x)
    X_trans[:,0] = x[:,0]**2
    X_trans[:,1] = x[:,1]**2
    X_trans[:,2] = x[:,2]
    x_T_c = np.dot(X_trans,c) + np.random.normal(0,1,1000)
    #plt.scatter(X_trans[:,1], x_T_c)
    #f = np.exp(x_T_c) 
    return x_T_c#, f

mu = np.array([0,0.1,0.3])
#Covariance array
cov_m_r = np.array([[1,0.94,0.01],[0.94,1,0], [0.01,0,1]])
corr_data = np.random.multivariate_normal(mu, cov_m_r,(1000))

X_df = pd.DataFrame(corr_data)


#Scaling data
scaler = StandardScaler()
X = scaler.fit_transform(X_df)
    

pca = PCA()
#print(corr_data)
pca.fit(X)

#Scree Plot of Principal Components
PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.ylim([0,1])
plt.show()
#Finding principal component matrix
A = pca.components_

#Projecting data onto principal components
A_proj = proj(A,X)
#Transpose projected data
X = A_proj.T


#Generating random data through model f
c = (0.8, 0.3, 0.01)
[x_t_c_1, f_1] = exp_f(c,X)
y_1 = np.log(f_1)

[x_t_c_2, f_2] = exp_f_2(c,X)
y_2 = np.log(f_2)

reg_1 =  LinearRegression().fit(X,y_1)
print('Regression Score', reg_1.score(X,y_1))

reg_2 =  LinearRegression().fit(X,y_2)
print('Regression Score', reg_2.score(X,y_2))

OOS_Data = np.random.multivariate_normal(mu, cov_m_r,(1000))


