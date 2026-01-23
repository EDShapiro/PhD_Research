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
    
    with open(net_dict_path, "rb") as file:
        data_dict = pickle.load(file)
        
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

    model = tf.keras.models.load_model(model_path, custom_objects={"EM_Noise_Augmentation": EM1DNoise})
    #Setting noise flag in custom layer to false
    model.layers[0].set_noise_flag(tf.cast(False, dtype = tf.bool))
    
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


#depth = discrete depth file for survey 
depth_path = f'./Data/simulated_data/date_10_10_25/depth_data.npy'
depth = import_depth(depth_path)

#Set in_data_path to original data path
# out_data_path = f'./Data/experiments/date_12_23_25/test_wn/tvt_data/y_val_data.npy'

# #Set in_data_path to processed data path
# in_data_path = f'./Data/experiments/date_12_23_25/test_wn/tvt_data/x_val_proc.npy'

# #Import CNN dictionary
# net_dict_path =   f'./Data/experiments/date_12_23_25/test_wn/tvt_data/net_dict.pkl'

# # Add path to model file
# model_path = f'./Data/experiments/date_12_23_25/test_wn/model/checkpoints/model_checkpoint_316.model.keras'


out_data_path = f'./Data/experiments/date_12_23_25/test_3_wn/tvt_data/y_train_data.npy'

#Set in_data_path to processed data path
in_data_path = f'./Data/experiments/date_12_23_25/test_3_wn/tvt_data/x_train_proc.npy'

#Import CNN dictionary
net_dict_path =   f'./Data/experiments/date_12_23_25/test_3_wn/tvt_data/net_dict.pkl'

# Add path to model file
# model_path = f'./Data/experiments/date_12_23_25/test_3_wn/model/checkpoints/model_checkpoint_303.model.keras'


model_path = f'./Data/experiments/date_12_23_25/test_3_wn/model/nn_model.keras'
# scaler_in_path = f'./Data/experiments/date_12_23_25/test_wo_n/tvt_data/scaler_in.pkl'
scaler_in_path = []
# mc_n = # of MC samples for inference
mc_n = 1000
plot_idx = 2
#
x_temp = np.load(in_data_path)[plot_idx,:]
height = x_temp[0]
del x_temp
idx_arr = np.repeat(plot_idx, mc_n )

# test_idx = resistivity model performing inference on
 


#Add height to plot

# training boolean set to False for deterministic predictioncd
# set True for stochastic prediction via stochastic dropout 
training = False
cust_drop = False
y_p_stoch = dropout_pred_fun(net_dict_path, model_path, in_data_path, idx_arr,
                             training, cust_drop, scaler_in_path)
cust_drop = False
training = False

# scaler_in_path = './Data/experiments/date_10_5_25/noise_005_test/tvt_data/scaler_in.pkl'
y_p_det = dropout_pred_fun(net_dict_path, model_path, in_data_path, idx_arr,
                           training, cust_drop, scaler_in_path )

pred_stoch_col = y_p_stoch[:,1:].flatten()
pred_det_col = y_p_det[:,1:].flatten()

#load true resistivity model
y_true = load_true_data(out_data_path, idx_arr)
true_res_col = y_true[:,1:].flatten()

res_col  = np.append(true_res_col, pred_stoch_col)
res_col = np.append(res_col, pred_det_col)

y_p_mean = np.mean(y_p_stoch[:,1:], axis = 0)
mrae_stoch = np.round(mrae_calc(y_true[0,1:],  y_p_mean),3)
mrae_det = mrae_calc(y_true[0,1:], y_p_det[0,1:])                
print('Stochastic Mean -MRAE = ', mrae_stoch)
print('Deterministic - MRAE = ', mrae_det)

label_temp = [ 'True Resistivity','CNN Mean Inverted Resistivity','CNN Deterministic Inverted Resistivity' ]
labels = np.repeat(label_temp, true_res_col.shape[0])

# pdepth = pre_proc_depth(depth)
depth_temp = np.repeat([depth], mc_n*3, axis = 0)

depth_col = depth_temp.flatten()

exp_dict = {'resistivity': res_col, 'Legend':labels, 'depth': depth_col}

plot_df = pd.DataFrame(exp_dict)

plt.figure(0)
sns.relplot(
    data=plot_df, kind="line",
    x='depth', y='resistivity', style = 'Legend', errorbar =("sd", 1.96), estimator = np.mean
)

# Show the plot
plt.yscale('log')
plt.ylim(0.5*true_res_col.min(),1.5*true_res_col.max())
plt.title(f'True Resistivity vs Dist. of CNN Inverted Resistivity - Test Data, MRAE = {mrae_stoch}, plot_idx = {plot_idx}, height = {height}')
plt.show()


