# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 22:41:31 2024

@author: shap293
"""

import datetime
import json
import numpy as np
import sklearn
import pickle
import os
import tensorflow as tf
from tensorflow import keras
import time
import joblib
import random
from numpy.random import seed
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv1D
from tensorflow.keras.constraints import max_norm
import matplotlib.pyplot as plt
# Instructions:
# Set the current working directory to the Prediction folder directory.
# Variables:
# plot_idx - Takes on values 1,...,N. Controls which resistivity profile is 
# plotted from the set of test profiles generated when Testing_Script.py is run.


def keras_prep(X, n_inputs, n_features):
    
   
    if X.ndim ==1:
        X_f = X.reshape((1, 11, 1))
    else:
        dim_0 = X.shape[0]
        X_f = X.reshape((dim_0, X.shape[1], 1))
    return X_f

def inc_data_prep(X, net_dict_path):
    data_dict = np.load(net_dict_path)
    n_inputs = data_dict['inputs_n']
    n_features = data_dict['features_n']
    #Convert data into array here?
    #X = np.array(X)
    X_f = X.reshape(n_features,  n_inputs)
    return X_f

    
def Res_Inv_Trans(Y, Y_l_max, Y_l_min):
    Y_inv = np.exp(Y*(Y_l_max - Y_l_min)+Y_l_min)
    return Y_inv


def new_data_pred(net_dict_path, in_data_path, out_data_path, model_path,
                  test_idx , input_type, test_type,
                  model_file, y_data_type, training, X_in = [], sample_type =[]):
    
    #Load dictionary with training parameters
    data_dict = np.load(net_dict_path)
    n_inputs = data_dict['inputs_n']
    n_features = data_dict['features_n']
    Y_l_max = data_dict['Y_l_max']
    Y_l_min = data_dict['Y_l_min']
    model = tf.keras.models.load_model(model_path)
        
    if input_type == 'file_1_data_set':
        X = np.load(in_data_path)
        X_p = X[test_idx,:]
        Y_p = model(X_p, training = training)
        Y_p = Res_Inv_Trans(Y_p, Y_l_max, Y_l_min)
        if test_type == 'y_known':
            if y_data_type == '.txt':
                Y = np.loadtxt(out_data_path, delimiter = ',')
                Y = Y.T
                Y_true = Y[test_idx,:]
            if y_data_type == '.npy':
                Y = np.load(out_data_path)
                Y_true = Y[test_idx,:]
            return [Y_true, Y_p]
        
        else:
            return [Y_p]
    
    if input_type == 'file_2_data_set':
        X_1 = np.load(in_data_path[0])
        X_2 = np.load(in_data_path[1])
        X_p_1 = X_1[test_idx,:]
        X_p_2 = X_2[test_idx,:]
        Y_p_1 = model.predict(X_p_1)
        Y_p_2 = model.predict(X_p_2)
        Y_p_1 = Res_Inv_Trans(Y_p_1, Y_l_max, Y_l_min)
        Y_p_2 = Res_Inv_Trans(Y_p_2, Y_l_max, Y_l_min)
        if test_type == 'y_known':
            Y = np.loadtxt(out_data_path, delimiter = ',')
            Y = Y.T
            Y_true = Y[test_idx,:]
            return [Y_true, Y_p_1, Y_p_2]
        else:
            return [Y_p_1, Y_p_2]
        
    if input_type == 'inc_data':
        # print(X_in)
        # X_test = inc_data_prep(X_in, net_dict_path)
        X_n = scaler_in.transform(X_in)
        X_p = keras_prep(X_n, n_inputs, n_features)
        Y_p = model.predict(X_p)
        Y_p = Res_Inv_Trans(Y_p, Y_l_max, Y_l_min)
        return [Y_p]
    

def import_depth(depth_path):
    depth_data = np.load(depth_path)
    # depth_data = []
    # for i in range(0,s_rmd_thk.shape[0]):
    #     depth_data = np.append(depth_data, 1/3*np.sum(s_rmd_thk[0:i+1]))
    return depth_data 
    
def rse_calc(Y_t, Y_p):
    Y_t_avg = Y_t.mean()
    rse = np.dot(Y_p - Y_t, Y_p - Y_t)/np.dot(Y_t - Y_t_avg, Y_t - Y_t_avg)
    return rse

def nrmse_calc(Y_t, Y_p):
    return  np.sqrt(np.round(np.dot(Y_p - Y_t, Y_p - Y_t)/np.dot(Y_t, Y_t),3))

def mrae_calc(Y_t, Y_p):
    return  np.mean(np.divide(np.abs(Y_t - Y_p), Y_t))

def mse_calc(Y_t, Y_p):
    return  np.round(1/Y_p.shape[0]*np.dot(Y_p - Y_t, Y_p - Y_t),3)

def n_std_err_calc(Y_t, Y_p):
    return  np.sqrt(np.round(np.dot(Y_p - Y_t, Y_p - Y_t)/np.dot(Y_t, Y_t),3))/Y_t.shape[0]

def drop_plot(y_t, y_p, depth, c_lev, labels, title):
    if y_t.shape[1] == 31:
        y_t = y_t[:,1:]
    if y_p.shape[1] == 31:
        y_p = y_p[:,1:]
    
    depth = np.repeat([depth], y_t.shape[0]*2, axis = 0)
    depth_col = depth.flatten()
    
    
    
    y_t = y_t.flatten()
    y_p = y_p.flatten()
    res_col  = np.append(y_t, y_p)
   
    
    labels = np.repeat(labels, y_t.shape[0])  
    
    exp_dict = {'resistivity': res_col, 'Legend':labels, 'depth': depth_col}
   
    plot_df = pd.DataFrame(exp_dict)

    sns.relplot(
        data = plot_df, kind="line",
        x='depth', y='resistivity', style = 'Legend', errorbar =("sd", c_lev), estimator = np.mean
    )
    
    plt.yscale('log')
    plt.ylim(0.5*y_t.min(),1.5*y_t.max())
    plt.title(title)
    plt.show()
    return
    
def res_plot(Y, plot_type, depth_path, plot_idx, title, label_list = []):
    depth = import_depth(depth_path)
    
    if plot_type == 'y_known_1_input':
        
        Y_t = Y[0]
        # Y_t = Y_t[plot_idx,:]
        Y_p = Y[1]
        # Y_p = Y_p[plot_idx,:]
        
        pdepth = np.repeat(np.r_[depth], 2)
        pdepth[:-1] = pdepth[1:]
        pdepth[-1] = 2*depth[-1]
        pres = np.repeat(Y_t, 2)
        pres_p = np.repeat(Y_p, 2)
        
        # Create figure
        fig = plt.figure(figsize=(7, 5), facecolor='w')
        fig.subplots_adjust(wspace=.25, hspace=.4)
             
            
        # abs_err = np.repeat(np.abs(np.subtract(Y_t, Y_p)),2)
            
        # Plot Resistivities
        ax1 = plt.subplot(1, 2, 1)
        plt.plot(pres, pdepth, 'k', label =  label_list[0])
        plt.plot(pres_p, pdepth, 'r', label = label_list[1])
        
        # plt.fill_between(abs_err, pdepth, 0, alpha=0.4)
        # plt.plot(abs_err, pdepth)
        plt.legend( fontsize = 9, draggable = True)
        plt.xscale('log')
        # plt.xlim([1e-1*Y_t[1:].min(), 1.5*Y_t[1:].max()])
        plt.ylim([1.5*depth[-1],0])
        plt.ylabel('Depth (m)')
        plt.xlabel(r'Resistivity $\rho_h\ (\Omega\,\rm{m})$')
        plt.title(title)
        plt.show
        
    if plot_type == 'y_known_2_input':
        
        
        Y_t = Y[0]
        # Y_t = Y_t[plot_idx,:]
        Y_p_1 = Y[1]
        # Y_p_1 = Y_p_1[plot_idx,:]
        Y_p_2 = Y[2]
        # Y_p_2 = Y_p_2[plot_idx,:]
        # 
       
        pdepth = np.repeat(np.r_[ depth], 2)
        pdepth[:-1] = pdepth[1:]
        pdepth[-1] = 2*depth[-1]
        pres = np.repeat(Y_t, 2)
        pres_p_1 = np.repeat(Y_p_1, 2)
        pres_p_2 = np.repeat(Y_p_2, 2)
        
        
        fig = plt.figure(figsize=(7, 5), facecolor='w')
        fig.subplots_adjust(wspace=.25, hspace=.4)
        
        # Create figure
        fig = plt.figure(figsize=(7, 5), facecolor='w')
        fig.subplots_adjust(wspace=.25, hspace=.4)
        # Plot Resistivities
        ax1 = plt.subplot(1, 2, 1)
        plt.plot(pres, pdepth, 'k', label =  label_list[0])
        plt.plot(pres_p_1, pdepth, 'r', label = label_list[1])
        plt.plot(pres_p_2, pdepth, 'g', label = label_list[2])
        plt.legend( fontsize = 9, draggable = True)
        plt.xscale('log')
        plt.xlim([0, 3000])
        plt.ylim([1.5*depth[-1],0])
        plt.ylabel('Depth (m)')
        plt.xlabel(r'Resistivity $\rho_h\ (\Omega\,\rm{m})$')
        plt.title(title)
        plt.show
        
    if plot_type == 'y_unknown':
            
        
        Y_p = Y[0]
        Y_p = Y_p[plot_idx, 1:]
        pdepth = np.repeat(np.r_[depth], 2)
        pdepth[:-1] = pdepth[1:]
        pdepth[-1] = 2*depth[-1]
        pres_p = np.repeat(Y_p, 2)
        
        # Create figure
        fig = plt.figure(figsize=(7, 5), facecolor='w')
        fig.subplots_adjust(wspace=.25, hspace=.4)
        
        # Plot Resistivities
        ax1 = plt.subplot(1, 2, 1)
        plt.plot(pres_p, pdepth, 'b', label = 'DL-RMD Database Model')
        plt.xscale('log')
        plt.xlim([0.75*Y_p[1:].min(), 1.5*Y_p[1:].max()])
        plt.ylim([1.5*depth[-1],-1])
        plt.ylabel('Depth (m)')
        plt.xlabel(r'Resistivity $\rho_h\ (\Omega\,\rm{m})$')
        plt.title(title)
        plt.show
        

    
