# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 21:39:14 2025

@author: Evan Shapiro
"""
import numpy as np

import random
import os
import sys

rng = np.random.default_rng() 
np_seed = 46 #rng.integers(low = 0, high = 1e7, size =1)
np.random.seed(np_seed)


# Add the path to the directory containing your module
# to the list of searchable Python paths (before importing it)

base_path = 'D:\\PNNl'
os.chdir(base_path)
module_dir = os.path.abspath('./Training/')
sys.path.insert(0, module_dir)
from Training_Function_Module import *

##Setting paths for training
# # #Necessary data paths

#EM Data save path
fEM_save_path = f'./Data/simulated_data/date_1_14_26/fEM_data.npy'

#resistivity data save path
rho_save_path = f'./Data/simulated_data/date_1_14_26/rho_data.npy'

depth_path =  f'./Data/simulated_data/date_1_14_26/depth_data.npy'
rho_data_path = f'./Data/simulated_data/date_1_14_26/rho_data_non_aug_norm.npy'
fem_data_path = f'./Data/simulated_data/date_1_14_26/fEM_data_non_aug_norm.npy'


#Save paths
tvt_save_path =  f'./Data/experiments/date_1_19_26/test_2_wn_norm/tvt_data/'
net_dict_path = "./Data/experiments/date_1_19_26/test_2_wn_norm/tvt_data/net_dict.pkl"

with open(sim_data_path, "rb") as file:
    x_temp = np.load(file)
    dim_0 = x_temp.shape[0]
    del x_temp
d_f = 1/5
N = int(d_f*dim_0)
# N = 1000

#If training a new model, set train_type to 'new'. If updating an existing model,
#set train type to 'update', and set
train_type = 'new'
#scaling method for fEM data
scaler_type = 'standard_scaler'
#N is  sample size.
#standard_scalar or min_max_scaler, or robust_scaler
# Number of input data channels for CNN
features_n = 1
# #off_n = number of receivers
off_n = 1
# #number of frequencies
freq_n = 5
#For multiple heights, inputs_n:= 1 height input and 10 inputs per receiver(offest)
#1 receiver means 11 inputs
#For single height, inputs_n = 2*n_off*n_freq 
#Number of original inputs
#If inputs_n = 2*off_n*freq_n 
inputs_n = 2*off_n*freq_n + 1 #adding 1 for height data

#Number of inputs per input feature for neural network
# If features_n = 1, cnn_inputs = inputs_n
# If features_n > 1, then cnn_inputs != inputs_n, and must be modified
# cnn_inputs = 2*off_n*freq_n + 1  

#Input data noise parameters    
#If noise == '0', then add  no Gaussian noise to data.
#If noise == '1',  then add Gaussian noise to data, with variance proportional
# of the point-wise amplitude of the field
noise = '1'

#noise_coef := Noise scale factor. Multiply the value defined in the noise option by the
#scale factor, take absolute value, multiple Gaussian noise by resulting value 
# noise_coef = 0.05

#Mean of noise distribution
noise_loc = 0

#Base standard deviation of proportional noise distribution
noise_std = 0.05

#Base standard deviation of floor noise distribution
floor_std = 0.01

#Training Validation Test Split
tvt_vals = [0.8, 0.1, 0.1]




if train_type == 'new':
    tts_dict = tvt_split(N, tvt_vals, dim_0, train_type)
    
    idx_dict = sample_data(tts_dict, fem_data_path, tvt_save_path, rho_data_path, 
                             idx_dict = [])

#Option for loading a previously trained model and training on new data
if train_type == 'update':
     cnn_dict = np.load(net_dict_load_path)
     train_idx = idx_dict['train_idx'].astype(int)
     test_idx = idx_dict['test_idx'].astype(int)
     val_idx = idx_dict['val_idx'].astype(int) 
     comp_idx = idx_dict['comp_idx'].astype(int)
     
     tts_dict = train_test_val_split(N, p_vals, dim_0, train_type, comp_idx[0,:])
     
     train_idx_int = np.append(train_idx[0,:], tts_dict['train_idx'])
     train_idx_arr = np.empty((n_heights, train_idx_int.shape[0]))
     train_idx_arr[0,:] = train_idx_int
     
     test_idx_int = np.append(test_idx[0,:], tts_dict['test_idx'])
     test_idx_arr = np.empty((n_heights, test_idx_int.shape[0]))
     test_idx_arr[0,:] = test_idx_int
     
     val_idx_int = np.append(val_idx[0,:], tts_dict['val_idx'])
     val_idx_arr = np.empty((n_heights, val_idx_int.shape[0]))
     val_idx_arr[0,:] = val_idx_int
     
     comp_idx_arr = np.empty((n_heights, tts_dict['c_idx'].shape[0]))
     comp_idx_arr[0,:] = tts_dict['c_idx'].astype(int)
     
     data_arr = sample_data( N, tvt_vals, sim_data_path, 
                       train_type, tvt_save_path, rho_path, depth_path, idx_dict = [])

net_dict = {'off_n':off_n, 'freq_n': freq_n, 'inputs_n': inputs_n, 'N':N,
            'features_n': features_n, 'noise': noise, 'noise_loc': noise_loc,
            'prop_scon':noise_std, "floor_scon": floor_std, 'sim_data_path': sim_data_path, 'depth_path': depth_path,
            'rho_path': rho_path, 'tvt_val': tvt_vals, 'sample_size': N, 
            'train_idx': idx_dict['train_idx'].astype(int),  'val_idx':  idx_dict['val_idx'].astype(int),
            'test_idx': idx_dict['test_idx'].astype(int), 'comp_idx': idx_dict['comp_idx'].astype(int)
            }

with open(net_dict_path, "wb") as file:
    pickle.dump(net_dict, file)
    
noise_scaling(tvt_save_path, net_dict_path)

pre_proc_res_data(net_dict_path, tvt_save_path)

 
