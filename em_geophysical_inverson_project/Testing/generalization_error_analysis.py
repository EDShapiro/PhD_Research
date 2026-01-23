# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 11:26:35 2025

@author: Evan Shapiro
"""

#Generalization error analysis
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 16:46:40 2025

@author: Evan Shapiro
"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import numpy

os.chdir('D:\\PNNL\\')
#Set path to the Testing folder in the cloned Github repository
module_dir = os.path.abspath('D:\\PNNL\\Testing\\')
sys.path.insert(0, module_dir)
module_dir = os.path.abspath('D:\\PNNL\\Training\\')
sys.path.insert(0, module_dir)

from Testing_Function_Module import *

def cust_inf_w_drop(model, inputs):
    """Run inference with Dropout active and BatchNorm in inference mode."""
    x = inputs
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            # Recursively handle nested models
            x = custom_inference_with_dropout(layer, x)

        elif isinstance(layer, tf.keras.layers.Dropout):
            x = layer(x, training=True)  # Force Dropout ON

        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            x = layer(x, training=False)  # Ensure BatchNorm uses inference mode

        else:
            x = layer(x)
    return x

def dropout_pred_fun(net_dict_path, model_path, in_data_path, test_idx, 
                     training, cust_drop = False, scaler_in_path = []):
    
    data_dict = np.load(net_dict_path) 
    inputs_n = data_dict['inputs_n']
    features_n = data_dict['features_n']
    Y_l_max = data_dict['Y_l_max']
    Y_l_min = data_dict['Y_l_min']
    off_n = data_dict['off_n']
    freq_n = data_dict['freq_n']
    
    X = np.load(in_data_path)
    X = X[test_idx,:]
    
    if X.shape[0] == 1: #reshape if single observation
        X = X.reshape(-1,1)
    
    if scaler_in_path:
        scaler_in = joblib.load(scaler_in_path)
        X_p = scaler_in.transform(X)
    else:
        X_p = X
    
    X_p = keras_prep(X_p, inputs_n, features_n)

    model = tf.keras.models.load_model(model_path)
        
    st = time.time()
    if cust_drop:
        Y_p = cust_inf_w_drop(model, X_p)
    else:
        Y_p = model(X_p, training = training)
    et = time.time()
    
    # Y_p = Res_Inv_Trans(Y_p, Y_l_max, Y_l_min)
    Y_p = 10**(Y_p*(Y_l_max - Y_l_min)+Y_l_min)
    # Y_p = 10**(Y_p)
    Y_p = Y_p.numpy()
    
    return Y_p

def load_true_data(out_data_path, idx_arr):
    Y = np.load(out_data_path)
    Y_true = Y[idx_arr,:]
    return Y_true

def add_noise_vec(x_path, x_idx, mod_sdev):
    X = np.load(x_path)
    
    x_vec = X[x_idx, :]
    
    x_vec_n = np.copy(x_vec)
    x_vec_n[1:] += np.multiply(np.abs(x_vec_n[1:]), np.random.normal(loc = 0, scale = mod_sdev, size = x_vec_n[1:].shape[0]))
    
    x_vec = np.array([x_vec])
    x_vec_n = np.array([x_vec_n])
    
    return x_vec, x_vec_n

def log_tr_data(x):
    return np.log(np.abs(x) + np.abs(x.min())*1e-5 )

def scale_input_data(X, scaler_in_path, net_dict_path):
    
    data_dict = np.load(net_dict_path) 
    inputs_n = data_dict['inputs_n']
    features_n = data_dict['features_n']
    
    scaler_in = joblib.load(scaler_in_path)
    X_p = scaler_in.transform(X)
    X_p = keras_prep(X_p, inputs_n, features_n)
    
    return X_p



def load_input_data(x_path, scaler_in_path, net_dict_path, x_idx, mc_n, mod_sdev):
    
    x_vec, x_vec_n = add_noise_vec(x_path, x_idx, mod_sdev)
    
    # x_ltr = log_tr_data(x_vec)
    # x_ltr_n = log_tr_data(x_vec_n)
    
    x_arr = np.repeat(x_vec, repeats = mc_n, axis = 0)
    x_arr_n = np.repeat(x_vec_n, repeats = mc_n, axis = 0)
    
    x_proc = scale_input_data(x_arr, scaler_in_path, net_dict_path)
    x_proc_n = scale_input_data(x_arr_n, scaler_in_path, net_dict_path)
    
    return x_proc, x_proc_n

def pred_fun(X_p, net_dict_path, model_path, cust_drop = False):
    
    data_dict = np.load(net_dict_path) 
    Y_l_max = data_dict['Y_l_max']
    Y_l_min = data_dict['Y_l_min']
    
    model = tf.keras.models.load_model(model_path) 
    st = time.time()
    if cust_drop:
        Y_p = cust_inf_w_drop(model, X_p)
    else:
        Y_p = model(X_p, training = training)
    et = time.time()
    
    # Y_p = Res_Inv_Trans(Y_p, Y_l_max, Y_l_min)
    Y_p = 10**(Y_p*(Y_l_max - Y_l_min)+Y_l_min)
    # Y_p = 10**(Y_p)
    Y_p = Y_p.numpy()
    return Y_p

#depth = discrete depth file for survey 
depth_path = f'./Data/RMD_Data/depth_data.npy'
depth = import_depth(depth_path)

#Set in_data_path to original data path
out_data_path = './Data/experiments//date_8_14_25/noise_05//tvt_data/y_test_data.npy'

#Set in_data_path to processed data path
in_data_path = './Data/experiments//date_8_14_25/noise_05//tvt_data/x_test_data.npy'

#Import CNN dictionary
net_dict_path =   './Data/experiments//date_8_14_25/noise_05//model/nn_dict.npz'

# Add path to model file
model_path = './Data/experiments//date_8_14_25/noise_05//model/nn_model.keras'

scaler_in_path = './Data/experiments//date_8_14_25/noise_05//tvt_data/scaler_in.pkl'

#path where error analysis arrays are saved
arr_save_path = './Data/experiments//date_8_14_25/noise_05//err_analysis/'

# scaler_in_path = []
# mc_n = # of MC samples for inference
mod_sdev = 0.05
mc_n = 1000
N = int(1e4)

mse_arr = np.zeros(N)
mrae_arr = np.zeros(N)

mse_arr_n = np.zeros(N)
mrae_arr_n = np.zeros(N)

idx_arr = np.linspace(0,N-1, N).astype(int)
idx_arr = np.random.choice(idx_arr, size = N, replace = False)

for i in range(0,N):
    data_idx = idx_arr[i]
    x_arr, x_arr_n = load_input_data(in_data_path, scaler_in_path,net_dict_path, data_idx, mc_n, mod_sdev)
    # training boolean set to False for deterministic prediction
    # set True for stochastic prediction via stochastic dropout 
    
    training = True
    #Turn on cust_crop if Batch Normalization was used during training to turn off
    #batch normalization during inference
    # cust_drop = False
    
    y_p_stoch = pred_fun(x_arr, net_dict_path, model_path, cust_drop = False)
    y_p_stoch_n = pred_fun(x_arr_n, net_dict_path, model_path, cust_drop = False)
    
    y_p_stoch = y_p_stoch[:,1:]
    y_p_mean = np.mean(y_p_stoch, axis = 0)
    
    y_p_stoch_n = y_p_stoch_n[:,1:]
    y_p_mean_n = np.mean(y_p_stoch_n, axis = 0)
    
    # cust_drop = False
    training = False
    
    
    #load true resistivity model
    y_true = load_true_data(out_data_path, idx_arr[0])[1:]
    
    nrmse = nrmse_calc(y_true, y_p_mean)
    mse_arr[i] = nrmse
    
    nrmse_n = nrmse_calc(y_true, y_p_mean_n)
    mse_arr_n[i] = nrmse_n
    
    mrae = mrae_calc(y_true, y_p_mean)
    mrae_arr[i] = mrae
    
    mrae_n = mrae_calc(y_true, y_p_mean_n)
    mrae_arr_n[i] = mrae_n

np.save(arr_save_path + 'mrae_arr', mrae_arr )    
np.save(arr_save_path + 'mse_arr', mse_arr ) 

np.save(arr_save_path + 'mrae_arr_n', mrae_arr_n )    
np.save(arr_save_path + 'mse_arr_n', mse_arr_n )

