# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 14:05:09 2023

@author: shap293
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 18:45:44 2023

@author: shap293
"""
#Primary Neural Network Script
#Before running change the working directory to directory containing
#the PNNL_Internship folder
import sklearn
import scipy as sp
import numpy as np 
from numpy.random import seed
import tensorflow as tf
from tensorflow import keras
import random
import os
import sys

rng = np.random.default_rng() 
np_seed = 46 #rng.integers(low = 0, high = 1e7, size =1)
tf_seed = 46 #rng.integers(low = 0, high = 1e7, size =1)
np.random.seed(np_seed)
tf.random.set_seed(tf_seed)

# Add the path to the directory containing your module
# to the list of searchable Python paths (before importing it)

base_path = 'D:\\PNNl'
os.chdir(base_path)
module_dir = os.path.abspath('./Training/')
sys.path.insert(0, module_dir)

# Now you can import your module as usual
from Training_Function_Module import *
from CNN_Module import *

#Load paths: If updating a previously trained model with new data set train
# type to update, and set the correct loading and save paths.
# Otherwise set train type to 'new', and define the loading paths to the existing
# trained model files path and the save paths to the path you wish to save the
# updated model paths to
#Loading paths
net_dict_load_path =  ''
scaler_in_load_path = ''
model_load_path = ''
tvt_data_path = f'./Data/experiments/date_1_19_26/test_2_wn_norm/tvt_data/'
net_dict_path = "./Data/experiments/date_1_19_26/test_2_wn_norm/tvt_data/net_dict.pkl"
# ./Data/experiments/date_6_18_25/puzyrev_test/max_pool/models/nn_model.keras

with open(net_dict_path, "rb") as file:
    net_dict = pickle.load(file)
#Save paths

checkpoint_path = "./Data/experiments/date_1_19_26/test_2_wn_norm/model/checkpoints/model_checkpoint_{epoch:02d}.model.keras"
model_path = f'./Data/experiments/date_1_19_26/test_2_wn_norn/model/nn_model.keras'

#If training a new model, set train_type to 'new'. If updating an existing model,
#set train type to 'update', and set
train_type = 'new'

#off_n = number of receivers
off_n = 1
freq_n = 5
cnn_inputs = 2*off_n*freq_n + 1 

# Number of input data channels for CNN
features_n = 1
##Data Properties
# Set the number of simulated data sets to use. Each height corresponds to
# a different simulated set, where simulated fEM data is collected at a unique
# height. Min height = 1, max height = 27`
# heights_n = 1
  
#Modeling Procedure:= 'new':= training model from scratch 'update':= updating
#previously trained model by training on new data.

##CNN Model Architecture Parameters
nn_type = "vgg_cnn"

#Dropout layer
#dropout_rate
dr_rate = 0.1
sp_dr_rate = 0.1
#flag for testing new cnn_arch
# 'old' =: vgg_cnn architecture that has been tested and validated
# 'test' =: test new architecture
mod_arch = 'old'


k_base_n = 64
#CNN kernel parameters
k_size_1 = 3 #Initial CNN Layer Filter Training Size
k_size_2 = 2 #Following CNN Layers Filter Training Size

#Stride of CNN filter
cnn_str = 1

#CNN kernel/filter constraints
kern_const = 'max_norm'
kern_con_size = 2
dense_kc = 2
#CNN Pooling layer parameters
#pool_type := 'max_pool' uses max pooling, 'avg_pool' uses average pooling
pool_type = 'avg_pool'
pool_size = 2
pool_stride = 1

#PReLu Initial Parameterization - Consistent with literature
prel_init = 0.35

#Weight Initialization for CNN
w_init = 'He'

#Number of neurons in dense layers of CNN
neur_n = 512

#L2 regularization penalty used in CNN for model output
reg = 1.0
##Model Training Parameters

#Early Stopping Criteria
#Patience
pat = 50

#DL Optimization Hyperparameters
#Max number of data epochs to train over
epochs = 500

#Initial optimizer Learning Rate
learn_rate = 1e-5

#Setting metrics for model training
str_metrics = ['mean_absolute_error','mean_squared_error']
str_labels = [ 'MAE', 'MSE']
loss = 'mse'

batch_size = 128

# initial_learning_rate = 0.01
# decay_steps = 1000  # Number of steps after which the learning rate decays
# decay_rate = 0.9    # Factor by which the learning rate is multiplied

# Create the ExponentialDecay schedule
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=initial_learning_rate,
#     decay_steps=decay_steps,
#     decay_rate=decay_rate,
#     staircase=True  # Set to True for discrete decay steps, False for continuous
# )



opt = keras.optimizers.AdamW( learning_rate = learn_rate, clipnorm = 2)


net_dict_temp = {'prel_init': prel_init,'weight_init': w_init,
                 'k_base_n': k_base_n,'k_size_1':  k_size_1, 'k_size_2': k_size_2, 
                 'kernel_constraint_type':kern_const ,'kern_con_size': kern_con_size,
                 'pool_type':pool_type, 'pool_size': pool_size, 'pool_stride':   pool_stride, 
                 'neur_n': neur_n, 'opt': opt,  'dr_rate': dr_rate, 'reg': reg,
                 'learning_rate': learn_rate,  'loss': loss,
                 'str_metrics': str_metrics, 'patience': pat, 'epochs': epochs,
                 'batch_size': batch_size
}
                 
net_dict.update(net_dict_temp)

with open(net_dict_path,"wb") as file:
    pickle.dump(net_dict, file)
    
                          
history = main_cnn_funct(checkpoint_path, net_dict_path, model_path, 
                   tvt_data_path, model_load_path = [], net_dict_load_path = [],
                   scaler_in_load_path = []
                   )
print('Model trained and saved to', model_path)
             

