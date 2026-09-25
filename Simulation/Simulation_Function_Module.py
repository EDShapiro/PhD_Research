# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 18:37:45 2024

@author: shap293
"""

import numpy as np
import scipy as sp
import empymod
import time
import os

def scale_data(min_val, max_val, data_path):
    s_rmd_bin = np.load(data_path)
    s_rmd_bin_scale = s_rmd_bin*(max_val-min_val)/(s_rmd_bin.max() - s_rmd_bin.min()) + (min_val*s_rmd_bin.max() - max_val*s_rmd_bin.min())/(s_rmd_bin.max() - s_rmd_bin.min())
    return s_rmd_bin_scale

def rnd_h_set(N):
    h_set = np.random.uniform(-20,-5, N)
    h_set_rounded = [round(h, 3) for h in h_set]
    np.save("./Data/h_list.npy", h_set_rounded)
    return 

def imp_data(rho_data, depth_data):
    s_rmd_rho = np.loadtxt(rho_data, delimiter = ',')
    s_rmd_thk = np.loadtxt(depth_data, delimiter = ',')
    print('Resistivity and Thickness Data Loaded')
    #Creating column names of indices from 1 to # of profiles 
    return s_rmd_rho, s_rmd_thk

 
def sim_empymod_data(rho_data, inp_data, save_path, mod_type, save_data=False):
    
    
    freq = inp_data['freqtime'][0]
    n_freq = inp_data['freqtime'].shape[0]
    n_rec = inp_data['rec'][0].shape[0]
    
    # 1. Initialize array for fEM data heights
    fEMBG_arr = np.zeros(2 * n_freq * n_rec )
    # fEMBG_arr[0] = inp_data['rec'][2][0]

    # 2. Run the Simulation (Moved up so fEMBG exists before use)
    if mod_type == 'dipole':
        # inp_data = {
        #     'src': [0, 0, t_h], 
        #     'rec': [fx, fy, r_h], 
        #     'depth': depth,
        #     'freqtime': freq, 
        #     'ab': ab, 
        #     'xdirect': xdirect,
        #     'htarg': {'pts_per_dec': -1}, 
        #     'verb': verb
        # }
        
        fEMBG = empymod.dipole(**inp_data, res=rho_data)
        
        # Apply physics corrections
        if n_rec == 1:
            for k in range(n_freq):
                fEMBG[k] = 1j * 8 * np.pi**2 * 1e-7 * freq[k] * fEMBG[k]
        else:
            for k in range(n_freq):
                fEMBG[k, :] = 1j * 8 * np.pi**2 * 1e-7 * freq[k] * fEMBG[k, :]

    elif mod_type == 'ratio':
        # inp_data = {
        #     'src': [0, 0, t_h], 
        #     'rec': [fx, fy, r_h], 
        #     'depth': depth,
        #     'freqtime': freq, 
        #     'ab': ab, 
        #     'verb': verb, 
        #     'htarg': {'pts_per_dec': -1}
        # }
        fEMBG = empymod.model.ip_and_q(**inp_data, res=rho_data)
        
    # 3. Format the results into the output array
    fEM_real = fEMBG[0]
    fEM_imag = fEMBG[1]
    
    if n_freq > 1:
        for k in range(n_freq):
            for m in range(n_rec):
                # Calculate index: 1 (offset) + 2*m (receiver pair) + 2*n_rec*k (frequency block)
                idx_real = 1 + 2 * m + 2 * n_rec * k
                idx_imag = 2 + 2 * m + 2 * n_rec * k
                
                if n_rec == 1:
                    fEMBG_arr[idx_real] = fEM_real[k]
                    fEMBG_arr[idx_imag] = fEM_imag[k]
                else:
                    fEMBG_arr[idx_real] = fEM_real[k, m]
                    fEMBG_arr[idx_imag] = fEM_imag[k, m]
    else:
        fEMBG_arr[0] = fEM_real
        fEMBG_arr[1] = fEM_imag

    # 4. Finalize
    if save_data:
        np.save(save_path, fEMBG_arr)

    return fEMBG_arr
 

def Res_Freq_Split(h_list, data_path, n_r, save_data_path, rec_config):   
    
    for i in range(0,h_list.shape[0]):
        temp_arr = np.load(data_path + f'fEM_data_{i}h_3r.npy')
        if n_r ==1:
            temp_arr_0 = np.empty(shape = (temp_arr.shape[0], 11))
            temp_arr_0[:,0] = temp_arr[:,0]
            if rec_config[0] == 1:
                j = 1
            if rec_config[0] == 2:
                j = 3
            if rec_config[0] == 3:
                j = 5
            for k in range(0,5):
                temp_arr_0[:,2*k+1: 2*k+3] = temp_arr[:,j + k*6:j+2 + k*6]
            np.save( save_data_path + f'fEM_data_{i}h_1r_{rec_config[0]}', temp_arr_0)
            
        if n_r ==2:
            temp_arr_0 = np.zeros((temp_arr.shape[0], 21))
            temp_arr_0[:,0] = temp_arr[:,0]
            
            if rec_config[0] == 1 and rec_config[1] == 2 : #First and second receiver
                j = 1
            if rec_config[0] == 2 and rec_config[1] == 3: #Second and 3rd Receiver
                j = 3
            
            for k in range(0,5):
                temp_arr_0[:,4*k+1: 4*k+5] = temp_arr[:,j + k*6: j + 4 + k*6]
            np.save(save_data_path + f'fEM_data_{i}h_2r', temp_arr_0)
    print('fEM Data Files Succesfully Saved')
    
        
        
        
        

    
def convert_to_real(h_list):
    os.chdir('/people/shap293/Drone_DL_Project/Test/Data/frequency_data/3_R')
    
    for i in range(0, 26):
        fEM_temp = np.load(f'fEM_data_{i}h.npy')
        fEM_temp = np.real(fEM_temp)
        np.save(f'fEM_data_{i}h', fEM_temp)
        

    