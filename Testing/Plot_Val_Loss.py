# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 15:19:57 2024

@author: shap293
"""
import os
import sys
os.chdir('D:\\PNNL\\')
module_dir = os.path.abspath('./Testing/')
sys.path.insert(0, module_dir)
from Testing_Function_Module import *

def plt_val_loss(nn_dict_path, title):
    cnn_dict = np.load(nn_dict_path)
    val_loss = cnn_dict['val_loss']
    plt.scatter(np.linspace(1,val_loss.shape[0],val_loss.shape[0]), val_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Data MSE')
    plt.title(title)
    return

nn_dict_path = './Data/experiments/date_10_23_25/noise_05_test/model/nn_dict.npz'
title = 'Validation Loss For _ Model'
plt_val_loss(nn_dict_path, title)
#Set paths to predicted data dictionaries
