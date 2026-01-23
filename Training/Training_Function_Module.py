 # -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 13:38:30 2023

@author: shap293
"""
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv1D
from tensorflow.keras.constraints import max_norm
import datetime
import time
import joblib
import pickle
import json
import numpy as np
import os
import glob

from CNN_Module import *



def keras_prep(X, n_inputs, n_features):
    X_f = X.reshape((X.shape[0], X.shape[1], 1))
    return X_f

class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

def imp_rho_dep_data(rho_path, depth_path):
    s_rmd_rho = np.load(rho_path)
    depth_data = np.load(depth_path)
    return s_rmd_rho, depth_data

#Exponential learning rate scheduler with warm startup
def lr_scheduler(epoch, lr):
    
    
    if epoch < 6:
        
        lr_list = np.logspace(-5, -3, num = 5)
        lr = lr_list[epoch]
        
    if epoch  >=6:
        lr = lr*np.exp(-0.1)
        
    return lr


def tvt_split(N, p_vals, dim_0, train_type, comp_idx = 0):
    
    p_check = np.sum(p_vals)
    if p_check != 1:
        print("Invalid test, train, validation split values. Values must sum to 1")
        return [], [],[]
    #Randomly sampling data
    if train_type == 'new':
        idx_list = np.arange(0,dim_0)
    else:
        idx_list = comp_idx.astype(int)
    
    
    n_train = int(p_vals[0]*N)
    n_val = int(p_vals[1]*N)
    n_test = int(p_vals[2]*N)
    # np.random.seed(12)
    val_idx = np.random.choice(sorted(idx_list), n_val, replace=False).astype(int)
    idx_list_2 = np.setdiff1d(idx_list, val_idx)
    # np.random.seed(19)
    test_idx =  np.random.choice(sorted(idx_list_2), n_test, replace=False).astype(int)
    idx_list_3 = np.setdiff1d(idx_list_2, test_idx).astype(int)
    # np.random.seed(15)
    train_idx = np.random.choice(sorted(idx_list_3), n_train, replace=False).astype(int)
    
    #Compliment of idx list used for testing and training
    c_idx = np.setdiff1d(idx_list_3, train_idx).astype(int)

    idx_dict = {'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx,
                'c_idx': c_idx}
    return idx_dict

#Function to randomly compile data for training, validation, and testing
# Also returns dictionary of tvt indices for each height array, as well as the 
# complimentary set of indices. The complimentary indices can be used
# to updating the NN with data it has not yet seen.

def log_tr_arr(X):
    X_l = np.copy(X)
    X_l[:,1:] = np.log(np.abs(X[:,1:]))
    return X_l



def sample_data( tts_dict, sim_data_path, save_tvt_data, rho_path, idx_dict = []):
    
    train_idx_arr = tts_dict['train_idx'].astype(int)
    val_idx_arr = tts_dict['val_idx'].astype(int)
    test_idx_arr = tts_dict['test_idx'].astype(int)
    comp_idx_arr = tts_dict['c_idx'].astype(int)
    #Loading Data
    Y = np.load(rho_path)
    X = np.load(sim_data_path)
    
    X_train = X[train_idx_arr,:]
    X_val =X[ val_idx_arr ,:]
    X_test = X[test_idx_arr,:]
    
    Y_train = Y[train_idx_arr,:]
    Y_val = Y[val_idx_arr,:]
    Y_test = Y[test_idx_arr,:]

    
    #saving the test data for later use
    np.save(save_tvt_data + 'x_train_data.npy', X_train)
    np.save(save_tvt_data + 'y_train_data.npy', Y_train)
    np.save(save_tvt_data + 'x_val_data.npy', X_val)
    np.save(save_tvt_data + 'y_val_data.npy', Y_val)
    np.save(save_tvt_data + 'x_test_data.npy', X_test)
    np.save(save_tvt_data + 'y_test_data.npy', Y_test)
    
    idx_dict = {'train_idx': train_idx_arr, 'val_idx': val_idx_arr, 'test_idx': test_idx_arr, 'comp_idx': comp_idx_arr}
     
    return idx_dict

def log_tr_input(input_data_path, save_data_path):
    X = np.load(input_data_path)
    X[:,1:] = np.log(np.abs(X[:,1:]))
    np.save(save_data_path + 'fEM_log_tr', X )
    return 

def Res_Trans(Y, Y_l_max, Y_l_min):
    Y_l = (np.log10(Y) - Y_l_min )/(Y_l_max - Y_l_min)
    return Y_l

def Res_Inv_Trans(Y, Y_l_max, Y_l_min):
    Y_inv = np.exp(Y*(Y_l_max - Y_l_min)+Y_l_min)
    return Y_inv


def import_rho_depth(depth_path):
    s_rmd_thk = np.loadtxt(depth_path, delimiter = ',')
    depth_data = []
    for i in range(0,s_rmd_thk.shape[0]):
        depth_data = np.append(depth_data, 1/3*np.sum(s_rmd_thk[0:i+1]))
    return depth_data 

#Function to standardize the noise added
def noise_scaling(tvt_save_path, net_dict_path):
    
    x = np.load(tvt_save_path + 'x_train_data.npy')
    #prevent taking log(0)
    log_eps = 1e-12
    
    with open(net_dict_path, 'rb') as f:
       net_dict = pickle.load(f)
    
    net_dict = np.load(net_dict_path, allow_pickle=True)
    
    scon_1 = net_dict["prop_scon"]
    scon_2 = net_dict["floor_scon"]
    
    feat_mean = np.mean(x, axis = 0)
    feat_st_dev = np.std(x, axis = 0)  
    feat_med = np.median(x[:,1:], axis = 0)
    
    prop_sfac =  scon_1*np.divide(feat_st_dev[1:], np.abs(feat_mean[1:]))
    floor_sfac = scon_2*np.abs(feat_med)
    
    log_feat_mean = np.zeros(x.shape[1])
    log_feat_st_dev = np.zeros(x.shape[1])
    
    log_feat_mean[0] = np.mean(x[:,0], axis = 0)
    log_feat_st_dev[0] = np.std(x[:,0], axis = 0)  
    
    log_feat_mean[1:] = np.mean(np.sign(x[:,1:])*np.log10(np.abs(x[:,1:]) + log_eps), axis = 0)
    log_feat_st_dev[1:] = np.std(np.log10(np.abs(x[:,1:]) + log_eps), axis = 0)  
    
    # log_feat_mean = np.zeros(2)
    # log_feat_st_dev = np.zeros(2)
    
    # log_feat_mean[0] = np.mean(X_train[:,0])
    # log_feat_st_dev[0] = np.std(X_train[:,0])  
    
    
    # log_feat_mean[1] = np.mean(np.sign(X_train[:,1:])*np.log10(np.abs(X_train[:,1:]) + log_eps))
    # log_feat_st_dev[1] = np.std(np.log10(np.abs(X_train[:,1:]) + log_eps))  
    
    # x_int = np.zeros(x.shape)
    
    # x_int[:,0] = x[:,0]
    # x_int[:,1:] = np.sign(x[:,1:])*np.log10(np.abs(x[:,1:]) + log_eps)
    
    # log_feat_mean = np.mean(x_int)
    # log_feat_st_dev = np.std(x_int)  
    
    net_dict.update({'feat_mean': feat_mean, 'feat_st_dev': feat_st_dev,
                     'log_feat_mean': log_feat_mean, 'log_feat_st_dev': log_feat_st_dev,
                     'prop_sfac': prop_sfac, 'floor_sfac': floor_sfac})
    
    with open(net_dict_path, 'wb') as f:
        pickle.dump(net_dict, f)
    
    return

    
def pre_proc_res_data(net_dict_path, tvt_save_path):
                            
    with open(net_dict_path, 'rb') as f:
       net_dict = pickle.load(f)
    # inputs_n = net_dict['inputs_n']
    # features_n = net_dict['features_n']
    
    #load net_dict and laod everything
    
    X_train = np.load(tvt_save_path + 'x_train_data.npy')
    X_val = np.load(tvt_save_path + 'x_val_data.npy')
    X_test = np.load(tvt_save_path + 'x_test_data.npy')
    Y_train = np.load(tvt_save_path + 'y_train_data.npy')
    Y_val = np.load(tvt_save_path + 'y_val_data.npy')
    Y_test = np.load(tvt_save_path + 'y_test_data.npy')
       
    #Standardizing the resistivity data using log normalization used by Asif
    # et al in" DL-RMD: a geophysically constrained..."
    Y_l_max = np.max(np.log10(Y_train))
    Y_l_min =  np.min(np.log10(Y_train))
    #save in net_dict
    net_dict.update({"Y_l_max":Y_l_max,"Y_l_min": Y_l_min})
                    
    with open(net_dict_path, 'wb') as f:
        pickle.dump(net_dict, f)
    
    
    # #Normalize the resistivity data using normalization in log space
    Y_train_n = Res_Trans(Y_train, Y_l_max, Y_l_min)
    Y_val_n = Res_Trans(Y_val, Y_l_max, Y_l_min)
    Y_test_n = Res_Trans(Y_test, Y_l_max, Y_l_min)
    
    # #Using log scaling as used by Puzyrev
    # Y_test_n = np.log10(Y_test)
    # Y_train_n = np.log10(Y_train)
    # Y_val_n = np.log10(Y_val)
    
    np.save(tvt_save_path + 'y_test_proc.npy', Y_test_n)
    np.save(tvt_save_path + 'y_train_proc.npy', Y_train_n)
    np.save(tvt_save_path + 'y_val_proc.npy', Y_val_n)

    # X_train_p = keras_prep(X_train, inputs_n, features_n)
    # X_val_p = keras_prep(X_val, inputs_n, features_n)
    # X_test_p = keras_prep(X_test, inputs_n, features_n)
    
    X_train_p = tf.convert_to_tensor(X_train)
    X_val_p = tf.convert_to_tensor(X_val)
    X_test_p = tf.convert_to_tensor(X_test)
    
    X_train_p = tf.expand_dims(X_train_p, axis = -1)
    X_val_p = tf.expand_dims(X_val_p, axis = -1)
    X_test_p = tf.expand_dims(X_test_p, axis = -1)
    
    np.save(tvt_save_path + 'x_train_proc.npy', X_train_p)
    np.save(tvt_save_path + 'x_val_proc.npy', X_val_p)
    np.save(tvt_save_path + 'x_test_proc.npy', X_test_p)
    return

""" Deprecated"""
def load_data_old(noise, noise_loc, noise_std,
                            scaler_type,  train_type,
                            inputs_n, features_n, off_n, freq_n,
                            tvt_save_path
                            ):
    
    
    X_train = np.load(tvt_save_path + 'x_train_data.npy')
    X_val = np.load(tvt_save_path + 'x_val_data.npy')
    X_test = np.load(tvt_save_path + 'x_test_data.npy')
    Y_train = np.load(tvt_save_path + 'y_train_data.npy')
    Y_val = np.load(tvt_save_path + 'y_val_data.npy')
    Y_test = np.load(tvt_save_path + 'y_test_data.npy')
    
    #Noise Model
    #Noise scale is proportional to the amplitude of the individual signal
    if noise == '1':
        
        X_train_noise = np.copy(X_train)
        X_test_noise = np.copy(X_test)
        X_val_noise = np.copy(X_val)
        
        #Setting proportional noise scaling factors for each feature except height
        feat_mean = np.mean(X_train[:,1:], axis = 0)
        feat_st_dev = np.std(X_train[:,1:], axis = 0)  
        sc_f =  np.divide(feat_st_dev, np.abs(feat_mean))
        
        feat_med = np.median(X_train[:,1:])
        
        
        np.save(tvt_save_path + 'data_sc_f.npy', sc_f)
        
        np.random.seed(42)
        X_train_noise[:,1:] += np.multiply(np.multiply(sc_f, np.abs(X_train_noise[:,1:])), np.random.normal(0, noise_std, size = X_train_noise[:,1:].shape))
        np.random.seed(24)
        X_val_noise[:,1:] += np.multiply(np.multiply(sc_f, np.abs(X_val_noise[:,1:])), np.random.normal(0, noise_std, size = X_val_noise[:,1:].shape))
        np.random.seed(4242)
        X_test_noise[:,1:] += np.multiply(np.multiply(sc_f, np.abs(X_test_noise[:,1:])), np.random.normal(0, noise_std, size = X_test_noise[:,1:].shape))
        

        # noise_coef_mat = np.abs(X_train_noise)
        # X_train_noise = X_train_noise + np.multiply(noise_coef_mat, np.random.normal(loc = noise_loc, scale = noise_std, size = X_train_noise.shape))   
        
        # noise_coef_mat = np.abs(X_val_noise)
        # X_val_noise = X_val_noise + np.multiply(noise_coef_mat, np.random.normal(loc = noise_loc, scale = noise_std, size = X_val_noise.shape))   

        # noise_coef_mat = np.abs(X_test_noise)         
        # X_test_noise = X_test_noise + np.multiply(noise_coef_mat, np.random.normal(loc = noise_loc, scale = noise_std, size = X_test_noise.shape))   
        
        
        X_train_l = log_tr_arr(X_train_noise)
        X_val_l = log_tr_arr(X_val_noise)
        X_test_l = log_tr_arr(X_test_noise)
        
        #save tvt data
        np.save(tvt_save_path + 'x_train_noise.npy', X_train_noise)
        np.save(tvt_save_path + 'x_val_noise.npy', X_val_noise)
        np.save(tvt_save_path + 'x_test_noise.npy', X_test_noise)
        
        np.save(tvt_save_path + 'x_train_ltr_noise.npy', X_train_l)
        np.save(tvt_save_path + 'x_val_ltr_noise.npy', X_val_l)
        np.save(tvt_save_path + 'x_test_ltr_noise.npy', X_test_l)
    
    #Noise Model
    #Noise scale is proportional to the amplitude of the individual signal
    if noise == '1':
        X_train_noise = np.copy(X_train)
        feat_mean = np.mean(X_train[:,1:], axis = 0)
        feat_st_dev = np.std(X_train[:,1:], axis = 0)  
        sc_f =  noise_std*np.divide(feat_st_dev, np.abs(feat_mean))
        np.save(tvt_save_path + 'data_sc_f.npy', sc_f)
       
    #Standardizing the resistivity data using log normalization used by Asif
    # et al in" DL-RMD: a geophysically constrained..."
    Y_l_max = np.max(np.log10(Y_train))
    Y_l_min =  np.min(np.log10(Y_train))
    
    # #Normalize the resistivity data using normalization in log space
    Y_train_n = Res_Trans(Y_train, Y_l_max, Y_l_min)
    Y_val_n = Res_Trans(Y_val, Y_l_max, Y_l_min)
    Y_test_n = Res_Trans(Y_test, Y_l_max, Y_l_min)
    
    # #Using log scaling as used by Puzyrev
    # Y_test_n = np.log10(Y_test)
    # Y_train_n = np.log10(Y_train)
    # Y_val_n = np.log10(Y_val)
    
    
    np.save(tvt_save_path + 'y_test_proc.npy', Y_test_n)
    np.save(tvt_save_path + 'y_train_proc.npy', Y_train_n)
    np.save(tvt_save_path + 'y_val_proc.npy', Y_val_n)

    #Normalize the data output data using the min and max frequency values 
    #of the entire output array
    
    if scaler_type =='min_max_scaler':
        scaler_in = MinMaxScaler()
        
    if scaler_type =='standard_scaler':
        scaler_in = StandardScaler()
        
    if scaler_type == 'robust_scaler':
        scaler_in = RobustScaler()
        
    if scaler_type !='min_max_scaler' and scaler_type !='standard_scaler' and scaler_type != 'robust_scaler':
        return print('scaler_type must either be min_max_scaler or standard_scaler or robust_scaler')
    
    if noise == '0':
        if train_type == 'new':
            scaler_in = scaler_in.fit(X_train)
            X_train_n = scaler_in.transform(X_train)
            X_val_n = scaler_in.transform(X_val)
            X_test_n = scaler_in.transform(X_test)
            
            
        if train_type == 'update': 
            scaler_in = joblib.load(scaler_in_load_path)
            X_train_n = scaler_in.transform(X_train)
            X_val_n = scaler_in.transform(X_val)
            X_test_n = scaler_in.transform(X_test)   
   
    if noise == '1':
        
        if train_type == 'new':
            scaler_in = scaler_in.fit(X_train_l)
            X_train_n = scaler_in.transform(X_train_l)
            X_val_n = scaler_in.transform(X_val_l)
            X_test_n = scaler_in.transform(X_test_l)
            
            
        if train_type == 'update': 
            scaler_in = joblib.load(scaler_in_load_path)
            X_train_n = scaler_in.transform(X_train_noise)
            X_val_n = scaler_in.transform(X_val_noise)
            X_test_n = scaler_in.transform(X_test_noise) 
            
    joblib.dump(scaler_in, tvt_save_path + 'scaler_in.pkl')
    
    X_train_p = keras_prep(X_train_n, inputs_n, features_n)
    X_val_p = keras_prep(X_val_n, inputs_n, features_n)
    X_test_p = keras_prep(X_test_n, inputs_n, features_n)
    
    np.save(tvt_save_path + 'x_train_proc.npy', X_train_p)
    np.save(tvt_save_path + 'x_val_proc.npy', X_val_p)
    np.save(tvt_save_path + 'x_test_proc.npy', X_test_p)
    
    Loaded_Data =  [Y_l_max, Y_l_min]
    return Loaded_Data


def noise_mod(data_path, net_dict_path):
    X = np.load(data_path + 'x_train_data.npy')
    with open(net_dict_path, "rb") as file:
        net_dict = pickle.load(file)
        
    std_dev = net_dict['noise_scale']
    feat_mean = np.mean(X, axis = 0)
    feat_st_dev = np.std(X, axis = 0)  
    sc_fac = std_dev*np.divide(feat_st_dev[1:], np.abs(feat_mean[1:]))
    
    net_dict['sc_fac'] = sc_fac
    net_dict['feat_mean'] = feat_mean
    net_dict['feat_st_dev'] = feat_st_dev
    
    with open(net_dict_path, "wb") as file:
        pickle.dump(net_dict, file)
    return 
    


def train_trans(X, net_dict_path):
    net_dict = np.load(net_dict_path)
    sc_fac = net_dict['sc_fac']
    mean = net_dict['feat_mean']
    std_dev = net_dict['feat_st_dev']    
    
    X_n = np.copy(X)
    X_n[:,1:,:] += np.multiply(np.multiply(sc_fac, np.abs(X_n[:,1:,:])), np.random.normal(loc = 0, scale = 1, size = X_n[:,1:,:]))
    X_n[:,1:,:] = np.log10(X_n[:,1:,:])
    X_n = (X_n - mean)/std_dev
    return X_n    

    

def main_cnn_funct(checkpoint_path, net_dict_path, model_path, 
                   tvt_data_path, model_load_path = [], net_dict_load_path = [],
                   scaler_in_load_path = []
                   ):
    
   
    with open(net_dict_path, "rb") as file:
        net_dict = pickle.load(file)
        
    pat = int(net_dict['patience'])
    batch_size = int(net_dict['batch_size'])
    epochs = int(net_dict['epochs'])
    N = net_dict['N']
         
    
    #Load data
    X_train = np.load(tvt_data_path + 'x_train_proc.npy')
    X_val = np.load(tvt_data_path + 'x_val_proc.npy')
    X_test = np.load(tvt_data_path + 'x_test_proc.npy')
    Y_train = np.load(tvt_data_path + 'y_train_proc.npy')
    Y_val = np.load(tvt_data_path + 'y_val_proc.npy')
    
    #Callbacks
    #Checkpoint Callback
    model_checkpt_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath = checkpoint_path,
        monitor='val_loss',
        mode='min',
        save_weights_only = False,
        save_best_only = True)

    early_stop_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=pat)
    
    lr_schedule_cb = keras.callbacks.LearningRateScheduler(lr_scheduler)
    
    #Setting optimizer
    # opt = keras.optimizers.AdamW(learning_rate = learn_rate)
    
    #Compile Model
    model = create_model(net_dict_path)
    # model = vgg_cnn_1r(net_dict_path)
    
    #Fitting new model
    tic = time.time()
    history = model.fit(X_train, Y_train, batch_size = batch_size,  epochs=epochs,
                        validation_data = (X_val, Y_val), 
                        verbose=1, callbacks=[model_checkpt_cb, early_stop_cb, 
                                              PrintDot()])
    toc = time.time()   
    #Updating a previously trained model by first loading the trained models weights
    #into a model with the same architecture, and then continuing training of the model
    #weights with new data
    print('Time to fit CNN =', toc- tic, '[s]')
    
    #Change model save path to PNNL_Internship Directory Path
    
    model.save(model_path)
    model = keras.models.load_model(model_path)
    
    st = time.time()
    model.predict(X_train)
    model.predict(X_val)
    model.predict(X_test)
    et = time.time()
    
    print('Time to predict', N, 'samples=', et - st,'[s]')
    print('Average time per prediction=', (et - st)/N, '[s]')
    
    
    net_dict.update({'model_history':history, 'train_loss': history.history['loss'],
                   'val_loss':history.history['val_loss'], 'epoch': history.epoch
        })
     
    #Create 2 separate dictionaries:
    # 1) Model parameters dictionary 
    # 2) Data & trained model dictionary
    
    with open(net_dict_path, "wb") as file:
        pickle.dump(net_dict, file)
    print('Model trained')
    return history

# def old_main_cnn_funct(net_dict, mod_arch,  train_type, off_n, w_init, prel_init,
#                    k_base_n, k_size_1, k_size_2, kern_const, kern_con_size, dense_kc,
#                    cnn_str, dr_rate,sp_dr_rate, neur_n, opt, loss, str_metrics, str_labels,
#                    cnn_inputs, features_n, pool_size, pool_stride, pool_type,
#                    reg, epochs, learn_rate, pat, batch_size,
#                    checkpoint_path, net_dict_save_path, model_save_path, 
#                    proc_data_path, model_load_path = [], net_dict_load_path = [],
#                    scaler_in_load_path = []
#                    ):
#     #Load data
#     X_train_p = np.load(proc_data_path + 'x_train_proc.npy')
#     X_val_p = np.load(proc_data_path + 'x_val_proc.npy')
#     X_test_p = np.load(proc_data_path + 'x_test_proc.npy')
#     Y_train_n = np.load(proc_data_path + 'y_train_proc.npy')
#     Y_val_n = np.load(proc_data_path + 'y_val_proc.npy')
    
#     data_std_dev = np.load(proc_data_path + 'data_sc_f.npy')
#     #Callbacks
#     #Checkpoint Callback
#     model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#         filepath = checkpoint_path,
#         monitor='val_loss',
#         mode='min',
#         save_weights_only = False,
#         save_best_only = True)

#     early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=pat)
    
#     #Setting optimizer
#     # opt = keras.optimizers.AdamW(learning_rate = learn_rate)
    
#     #Compile Model
#     if off_n == 1 and mod_arch != 'test':
#         model = vgg_cnn_1r(w_init, prel_init, k_base_n, k_size_1, cnn_str, k_size_2, 
#                        kern_con_size, dr_rate, sp_dr_rate, neur_n, opt, loss, str_metrics, cnn_inputs,
#                        features_n, pool_type, pool_size, pool_stride, reg, data_std_dev)
        
#     if off_n >= 3 and mod_arch != 'test':
#         model = VGG_CNN(w_init, prel_init, k_base_n, k_size_1, cnn_str, k_size_2, 
#                            kern_con_size, dr_rate, neur_n, opt, loss, str_metrics, cnn_inputs,
#                            features_n, pool_size, pool_stride, reg)
        
#     if mod_arch == 'test':
#         model = cnn_test(w_init, prel_init, k_base_n, k_size_1, cnn_str, k_size_2, 
#                        kern_con_size, dense_kc, dr_rate,  sp_dr_rate, neur_n, opt, loss, str_metrics, 
#                        cnn_inputs, features_n, pool_type, pool_size, pool_stride, reg)
                       
    
#     #Fitting new model
#     if train_type == 'new':
#         tic = time.time()
#         history = model.fit(X_train_p, Y_train_n, batch_size = batch_size,  epochs=epochs,
#                             validation_data = (X_val_p, Y_val_n),
#                             verbose=0, callbacks=[model_checkpoint_callback, early_stop, PrintDot()])
#         toc = time.time()
       
#     #Updating a previously trained model by first loading the trained models weights
#     #into a model with the same architecture, and then continuing training of the model
#     #weights with new data
#     if train_type == 'update':
#         model = keras.models.load_model(model_load_path)
#         tic = time.time()
#         history = model.fit(X_train_p, Y_train_n, batch_size = batch_size,  epochs=epochs,
#                             validation_data = (X_val_p, Y_val_n),
#                             verbose = 0, callbacks=[model_checkpoint_callback, early_stop, PrintDot()])
#         toc = time.time()
    
#     print('Time to fit CNN =', toc- tic, '[s]')
    
#     #Change model save path to PNNL_Internship Directory Path
    
#     model.save(model_save_path)
#     model = keras.models.load_model(model_save_path)
    
#     st = time.time()
#     model.predict(X_train_p)
#     model.predict(X_val_p)
#     model.predict(X_test_p)
#     et = time.time()
    
#     N = X_train_p.shape[0] +  X_val_p.shape[0] +  X_test_p.shape[0]
#     print('Time to predict', N, 'samples=', et - st,'[s]')
#     print('Average time per prediction=', (et - st)/(N), '[s]')
    
    
    
#     net_dict['model_history'] =  history
#     net_dict['train_loss'] = history.history['loss']
#     net_dict['val_loss'] = history.history['val_loss']
#     net_dict['epoch'] =  history.epoch
#     #Create 2 separate dictionaries:
#     # 1) Model parameters dictionary 
#     # 2) Data & trained model dictionary
    
#     np.savez_compressed(net_dict_save_path,**net_dict)
#     print('Model trained')
#     return history
#  #Option for training a new model

     

 
