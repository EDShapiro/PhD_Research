# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 00:33:36 2023

@author: shap293
"""
##Set of developed NN models

import tensorflow as tf
from tensorflow.keras import layers
from Training_Function_Module import *

#decorator so it is not necessary to load unique config() layer info each
# time a model is loaded

@tf.keras.utils.register_keras_serializable(package="EMProject")
class EM1DNoise(layers.Layer):
    def __init__(self, prop_sfac, floor_sfac, log_feat_mean, log_feat_st_dev, **kwargs):
        #load parent class information (layer.Layers)
        super().__init__(**kwargs)
        #Change code to load variables here from the net_dict, rather than passing them
        #Make noise_flag a passable variable
        self.epsilon = tf.cast(1e-11, dtype = tf.float32)
        self.prop_sfac = tf.reshape(tf.cast(prop_sfac, dtype = tf.float32), (1,-1,1)) 
        self.floor_sfac =  tf.reshape(tf.cast(floor_sfac, dtype = tf.float32), (1,-1,1)) 
        self._noise_flag = tf.Variable(True, trainable = False, dtype = tf.bool, name = "noise_flag")
        self.log_feat_mean = tf.reshape(tf.cast(log_feat_mean, dtype = tf.float32), (1,-1,1) )
        self.log_feat_st_dev = tf.reshape(tf.cast(log_feat_st_dev, dtype = tf.float32), (1,-1,1))
    
    #Defining getter and setter methods for noise_flag
    def get_noise_flag(self):
        return self._noise_flag
    
    def set_noise_flag(self, val):
        self._noise_flag = val
        return 
    
    def call(self, inputs, training = None):
        
        height_col = inputs[:,0:1,:]
        
        noise_col = inputs[:,1:,:]
        
        if training is None:
            training = False
        
        
        train_flag = tf.cast(training, dtype = tf.bool) 
        
        
        def apply_noise():
        
        # if noise_flag == True:
            #test this to see what the noise value are
            noise_shape = tf.shape(noise_col)
            
            noise_prop = self.prop_sfac*tf.abs(noise_col)*tf.random.normal(shape = noise_shape)
            
            noise_floor = self.floor_sfac*tf.random.normal(shape = noise_shape)
            
            noise_data = noise_col + noise_prop + noise_floor
            
            sign_x = tf.math.sign(noise_data)
            
            abs_val = tf.math.maximum(tf.math.abs(noise_data), self.epsilon)
            
            log_data = sign_x*tf.math.log(abs_val)/tf.math.log(10.)
            
            return log_data
        
       
            
            
        
        def no_noise():
            
            sign_x = tf.math.sign(noise_col)
            
            abs_val = tf.math.maximum(tf.math.abs(noise_col), self.epsilon)
            
            log_data = sign_x*tf.math.log(abs_val)/tf.math.log(10.)
            
            return log_data
        
        # proc_data = tf.where(noise_flag, apply_noise(), no_noise())
        
        # comb_data = tf.concat([first_col, proc_data], axis = 1)
        
        # st_data = (comb_data - self.log_feat_mean)/(self.log_feat_st_dev + self.epsilon)
        
        # sign_x = tf.math.sign(noise_col)
        
        # proc_data = sign_x*tf.math.log(tf.abs(noise_col) + self.epsilon)/tf.math.log(10.)
        
        proc_data = tf.cond(
            tf.math.logical_and(self.get_noise_flag(), train_flag), 
            lambda: apply_noise(), 
            lambda: no_noise()       
            )
        
        # st_noise_data = (proc_data - self.log_feat_mean[0,1,0])/ (self.log_feat_st_dev[0,1,0] + self.epsilon)
        
        comb_data = tf.concat([height_col, proc_data], axis = 1)
        
        st_data = (comb_data - self.log_feat_mean)/(self.log_feat_st_dev)
        
        return st_data
    
    def get_config(self):
        
        def conv_for_ser(x):
            #check to see if object has numpy attribute 
            if hasattr(x, "numpy"):
                #if it does convert to numpy object
                x = x.numpy()
            if hasattr(x, "tolist"):
                x = x.tolist()
            else:
                x = float(x)
            return x
        """Allows Keras to save/load the layer parameters."""
        config = super().get_config()
        config.update({
            "prop_sfac": conv_for_ser(self.prop_sfac),
            "floor_sfac": conv_for_ser(self.floor_sfac),
            "log_feat_mean": conv_for_ser(self.log_feat_mean),
            "log_feat_st_dev": conv_for_ser(self.log_feat_st_dev),
            })
        return config

def create_model(net_dict_path):
    
    with open(net_dict_path, "rb") as file:
        net_dict = pickle.load(file)
        
    weight_init, prel_init, k_base_n = net_dict['weight_init'],float(net_dict['prel_init']), int(net_dict['k_base_n'])
    k_size_1, k_size_2, kc = int(net_dict['k_size_1']),  int(net_dict['k_size_2']), int(net_dict['kern_con_size'])
    pool_type, n_pool, pool_stride = net_dict['pool_type'], int(net_dict['pool_size']), int(net_dict['pool_stride'])
    dr, n_neur, opt = float(net_dict['dr_rate']), int(net_dict['neur_n']), net_dict['opt']
    loss, str_metrics = str(net_dict['loss']), net_dict['str_metrics']
    prop_sfac, floor_sfac =  net_dict['prop_sfac'], net_dict['floor_sfac']
    log_feat_mean, log_feat_st_dev  = net_dict['log_feat_mean'], net_dict['log_feat_st_dev']
    # noise_flag = net_dict['noise_flag'] 
    # inputs_n, features_n = net_dict['inputs_n'], net_dict['features_n']
    reg = float(net_dict['reg'])
    
    if weight_init == 'He':
        init = tf.keras.initializers.HeNormal(seed = 42)
    if weight_init == 'Gl':
        init = tf.keras.initializers.GlorotNormal(seed = 42)
    
    # Helper for pooling logic
    def get_pooling_layer():
        if pool_type == 'avg_pool':
            return layers.AveragePooling1D(pool_size = (n_pool, ) , strides=(pool_stride,))
        elif pool_type == 'max_pool':
            return layers.MaxPooling1D(pool_size = (n_pool, ) , strides=(pool_stride,) )
        return None

    # 1. Define the layer list
    model = keras.Sequential( [
        layers.Input(shape=(11,1,)),
        EM1DNoise(prop_sfac=prop_sfac, floor_sfac=floor_sfac, log_feat_mean=log_feat_mean, 
                  log_feat_st_dev=log_feat_st_dev, name="EM_Noise_Augmentation"),

        # First set of convolutional layers (strides as 1-tuple)
        layers.Conv1D(filters=k_base_n, kernel_size=(k_size_1,), strides=(1,), 
                      kernel_initializer=init, padding="same", kernel_constraint=max_norm(kc)),
        layers.PReLU(alpha_initializer=tf.keras.initializers.constant(prel_init)),
        
        layers.Conv1D(filters=k_base_n, kernel_size=(k_size_1,), strides=(1,), 
                      kernel_initializer=init, padding="same", kernel_constraint=max_norm(kc)),
        layers.PReLU(alpha_initializer=tf.keras.initializers.constant(prel_init)),
        
        layers.Conv1D(filters=k_base_n, kernel_size=(k_size_1,), strides=(1,), 
                      kernel_initializer=init, padding="same", kernel_constraint=max_norm(kc)),
        layers.PReLU(alpha_initializer=tf.keras.initializers.constant(prel_init)),
        
        layers.Conv1D(filters=k_base_n, kernel_size=(k_size_1,), strides=(1,), 
                      kernel_initializer=init, padding="same", kernel_constraint=max_norm(kc)),
        layers.PReLU(alpha_initializer=tf.keras.initializers.constant(prel_init)),
        
        get_pooling_layer(),

        # Second set of convolutional layers (strides as 1-tuple)
        layers.Conv1D(filters=2*k_base_n, kernel_size=(k_size_2,), strides=(1,), 
                      kernel_initializer=init, padding="same", kernel_constraint=max_norm(kc)),
        layers.PReLU(alpha_initializer=tf.keras.initializers.constant(prel_init)),
        
        layers.Conv1D(filters=2*k_base_n, kernel_size=(k_size_2,), strides=(1,), 
                      kernel_initializer=init, padding="same", kernel_constraint=max_norm(kc)),
        layers.PReLU(alpha_initializer=tf.keras.initializers.constant(prel_init)),
        
        layers.Conv1D(filters=2*k_base_n, kernel_size=(k_size_2,), strides=(1,), 
                      kernel_initializer=init, padding="same", kernel_constraint=max_norm(kc)),
        layers.PReLU(alpha_initializer=tf.keras.initializers.constant(prel_init)),
        
        layers.Conv1D(filters=2*k_base_n, kernel_size=(k_size_2,), strides=(1,), 
                      kernel_initializer=init, padding="same", kernel_constraint=max_norm(kc)),
        layers.PReLU(alpha_initializer=tf.keras.initializers.constant(prel_init)),

        get_pooling_layer(),

        # Third set of convolutional layers (strides as 1-tuple)
        layers.Conv1D(filters=4*k_base_n, kernel_size=(k_size_2,), strides=(1,), 
                      kernel_initializer=init, padding="same", kernel_constraint=max_norm(kc)),
        layers.PReLU(alpha_initializer=tf.keras.initializers.constant(prel_init)),
        
        layers.Conv1D(filters=4*k_base_n, kernel_size=(k_size_2,), strides=(1,), 
                      kernel_initializer=init, padding="same", kernel_constraint=max_norm(kc)),
        layers.PReLU(alpha_initializer=tf.keras.initializers.constant(prel_init)),
        
        layers.Conv1D(filters=4*k_base_n, kernel_size=(k_size_2,), strides=(1,), 
                      kernel_initializer=init, padding="same", kernel_constraint=max_norm(kc)),
        layers.PReLU(alpha_initializer=tf.keras.initializers.constant(prel_init)),
        
        layers.Conv1D(filters=4*k_base_n, kernel_size=(k_size_2,), strides=(1,), 
                      kernel_initializer=init, padding="same", kernel_constraint=max_norm(kc)),
        layers.PReLU(alpha_initializer=tf.keras.initializers.constant(prel_init)),

        get_pooling_layer(),

        # Flatten and Dense layers
        layers.Flatten(),
        layers.Dropout(rate=0.1),
        layers.Dense(units=n_neur),
        layers.PReLU(alpha_initializer=tf.keras.initializers.constant(prel_init)),
        layers.Dropout(rate=dr),
        layers.Dense(units=n_neur),
        layers.PReLU(alpha_initializer=tf.keras.initializers.constant(prel_init)),
        layers.Dropout(rate=dr),
        layers.Dense(units=n_neur),
        layers.PReLU(alpha_initializer=tf.keras.initializers.constant(prel_init)),
        layers.Dropout(rate=dr),
        layers.Dense(units=n_neur),
        layers.PReLU(alpha_initializer=tf.keras.initializers.constant(prel_init)),
        layers.Dense(units=31, kernel_regularizer=tf.keras.regularizers.L2(reg))
    ])
    # model_layers = [l for l in model_layers if l is not None]

    # 3. Instantiate the model
    # model = keras.Sequential(model_layers)

    # Compile and return
    model.compile(optimizer=opt, loss=loss, metrics=  ['mean_absolute_error','mean_squared_error'])
    print(model.summary())
    return model


def vgg_cnn_1r(net_dict_path):
    
    net_dict = np.load(net_dict_path)
    weight_init, prel_init, k_base_n = net_dict['weight_init'], net_dict['prel_init'], net_dict['k_base_n']
    k_size_1, k_size_2, kc = net_dict['k_size_1'],  net_dict['k_size_2'], net_dict['kern_con_size']
    pool_type, n_pool, p_n_str = net_dict['pool_type'], net_dict['pool_size'], net_dict['pool_stride']
    dr, n_neur, opt = net_dict['dr_rate'], net_dict['neur_n'], net_dict['opt']
    loss, str_metrics = net_dict['loss'], net_dict['str_metrics']
    prop_sfac, floor_sfac =  net_dict['prop_sfac'], net_dict['prop_sfac']
    feat_mean, feat_st_dev  = net_dict['feat_mean'], net_dict['feat_st_dev']
    inputs_n, features_n = net_dict['inputs_n'], net_dict['features_n']
    
    if weight_init == 'He':
        init = tf.keras.initializers.HeNormal(seed = 42)
    if weight_init == 'Gl':
        init = tf.keras.initializers.GlorotNormal(seed = 42)
    
    model = keras.Sequential([ layers.Input(shape = (11,)),
                              EM1DNoise(prop_sfac = prop_sfac, floor_sfac = floor_sfac, feat_mean = feat_mean, feat_st_dev =feat_st_dev,
                                        name = "EM_Noise_Augmentation")
                              ] )
    
    #First set of convolutional layers
    model.add(keras.layers.Conv1D(filters = k_base_n, kernel_size = (k_size_1, ), strides=1,
        kernel_initializer = init, padding = "same", input_shape=( inputs_n, features_n), kernel_constraint=max_norm(kc)))
    model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))    
    # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
    model.add(keras.layers.Conv1D(filters=k_base_n, kernel_size=k_size_1, strides=1,
        kernel_initializer=init, padding="same", kernel_constraint=max_norm(kc)))
    model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
    model.add(keras.layers.Conv1D(filters=k_base_n, kernel_size=k_size_1, strides=1,
        kernel_initializer=init, padding="same", kernel_constraint=max_norm(kc)))
    model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))  
    # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
    model.add(keras.layers.Conv1D(filters=k_base_n, kernel_size=k_size_1, strides=1,
        kernel_initializer=init, padding="same", kernel_constraint=max_norm(kc)))
    model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))  
    # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
    # model.add(keras.layers.Conv1D(filters=k_base_n, kernel_size=k_size_1, strides=1,
    #     kernel_initializer=init, padding="same", kernel_constraint=max_norm(kc)))
    # model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))  
    # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
    #First set of convolutional layers
    # keras.layers.Conv1D(filters=k_base_n, kernel_size=k_size_1, strides=1,
    #     kernel_initializer=init, padding="same", input_shape=( n_inputs,n_features), kernel_constraint=max_norm(kc)),
    # keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)),    
    # # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
    # keras.layers.Conv1D(filters=k_base_n, kernel_size=k_size_1, strides=1,
    #     kernel_initializer=init, padding="same", kernel_constraint=max_norm(kc)),
    # keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)),
    # # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
    # keras.layers.Conv1D(filters=k_base_n, kernel_size=k_size_1, strides=1,
    #     kernel_initializer=init, padding="same", kernel_constraint=max_norm(kc)),
    # keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)),  
    # # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
    # keras.layers.Conv1D(filters=k_base_n, kernel_size=k_size_1, strides=1,
    #     kernel_initializer=init, padding="same", kernel_constraint=max_norm(kc)),
    # keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init))), 
    # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
    # model.add(keras.layers.Conv1D(filters=k_base_n, kernel_size=k_size_1, strides=1,
    #     kernel_initializer=init, padding="same", kernel_constraint=max_norm(kc)))
    # model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))  
    # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
    
   
    if pool_type == 'avg_pool':
        model.add(keras.layers.AveragePooling1D(pool_size=n_pool, strides=p_n_str))
    if pool_type == 'max_pool':
        model.add( keras.layers.MaxPooling1D(pool_size=n_pool, strides=p_n_str))
     
    #Second set of convolutional layers
    model.add(keras.layers.Conv1D(kernel_initializer=init, filters=2*k_base_n, kernel_size=k_size_2, strides=1, padding="same", kernel_constraint=max_norm(kc)))
    model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
    model.add(keras.layers.Conv1D(kernel_initializer=init, filters=2*k_base_n, kernel_size=k_size_2, strides=1, padding="same",kernel_constraint=max_norm(kc)))
    model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
    model.add(keras.layers.Conv1D(kernel_initializer=init, filters=2*k_base_n, kernel_size=k_size_2, strides=1, padding="same",kernel_constraint=max_norm(kc)))
    model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
    model.add(keras.layers.Conv1D(kernel_initializer=init, filters=2*k_base_n, kernel_size=k_size_2, strides=1, padding="same",kernel_constraint=max_norm(kc)))
    model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
    # model.add(keras.layers.Conv1D(kernel_initializer=init, filters=2*k_base_n, kernel_size=k_size_2, strides=1, padding="same",kernel_constraint=max_norm(kc)))
    # model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
    
    
    if pool_type == 'avg_pool':
        model.add(keras.layers.AveragePooling1D(pool_size=n_pool, strides=p_n_str))
    if pool_type == 'max_pool':
        model.add( keras.layers.MaxPooling1D(pool_size=n_pool, strides=p_n_str))
    
   
    #Third set of convolutional layers
    model.add(keras.layers.Conv1D(kernel_initializer=init, filters=4*k_base_n, kernel_size=k_size_2, strides=1, padding="same",kernel_constraint=max_norm(kc)))
    model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
    model.add(keras.layers.Conv1D(kernel_initializer=init, filters=4*k_base_n, kernel_size=k_size_2, strides=1, padding="same",kernel_constraint=max_norm(kc)))
    model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
    model.add(keras.layers.Conv1D(kernel_initializer=init, filters=4*k_base_n, kernel_size=k_size_2, strides=1, padding="same",kernel_constraint=max_norm(kc)))
    model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
    model.add(keras.layers.Conv1D(kernel_initializer=init, filters=4*k_base_n, kernel_size=k_size_2, strides=1, padding="same",kernel_constraint=max_norm(kc)))
    model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
    # model.add(keras.layers.Conv1D(kernel_initializer=init, filters=4*k_base_n, kernel_size=k_size_2, strides=1, padding="same",kernel_constraint=max_norm(kc)))
    # model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
    
     
    if pool_type == 'avg_pool':
        model.add(keras.layers.AveragePooling1D(pool_size=n_pool, strides=p_n_str))
    if pool_type == 'max_pool':
        model.add( keras.layers.MaxPooling1D(pool_size=n_pool, strides=p_n_str))
    
    # #4th set of convolutional layers
    # model.add(keras.layers.Conv1D(kernel_initializer=init, filters=8*k_base_n, kernel_size=k_size_2, strides=1, padding="same",kernel_constraint=max_norm(kc)))
    # model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    # # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
    # model.add(keras.layers.Conv1D(kernel_initializer=init, filters=8*k_base_n, kernel_size=k_size_2, strides=1, padding="same",kernel_constraint=max_norm(kc)))
    # model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    # # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
    # model.add(keras.layers.Conv1D(kernel_initializer=init, filters=8*k_base_n, kernel_size=k_size_2, strides=1, padding="same",kernel_constraint=max_norm(kc)))
    # model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    # # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
    # model.add(keras.layers.Conv1D(kernel_initializer=init, filters=8*k_base_n, kernel_size=k_size_2, strides=1, padding="same",kernel_constraint=max_norm(kc)))
    # model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    # # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
    # model.add(keras.layers.Conv1D(kernel_initializer=init, filters=8*k_base_n, kernel_size=k_size_2, strides=1, padding="same", kernel_constraint=max_norm(kc)))
    # model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    # # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
     
    # if pool_type == 'avg_pool':
    #     model.add(keras.layers.AveragePooling1D(pool_size=n_pool, strides=p_n_str))
    # if pool_type == 'max_pool':
    #     model.add( keras.layers.MaxPooling1D(pool_size=n_pool, strides=p_n_str))
        
    model.add(keras.layers.Flatten())
    
    ### the set of densely connected layers
    model.add(keras.layers.Dropout(rate = 0.1))
    # model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(units = n_neur))
    model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    model.add(keras.layers.Dropout(rate = dr ))
    # model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(units = n_neur))
    model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    model.add(keras.layers.Dropout(rate = dr ))
    # model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(units = n_neur))
    model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    model.add(keras.layers.Dropout(rate = dr ))
    # model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(units = n_neur))
    model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    # model.add(keras.layers.Dropout(rate = dr))
    # model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(units = 31, kernel_regularizer = tf.keras.regularizers.L2(reg)))
    
    ###Print Model Architecture
    print(model.summary())
    model.compile(optimizer=opt, loss= loss, metrics = str_metrics)
    return model    
    




# def vgg_cnn_1r(net_dict_path):
    
#     net_dict = np.load(net_dict_path)
#     weight_init, prel_init, k_base_n = net_dict['weight_init'], net_dict['prel_init'], net_dict['k_base_n']
#     k_size_1, k_size_2, kc = net_dict['k_size_1'],  net_dict['k_size_2'], net_dict['kern_con_size']
#     pool_type, n_pool, p_n_str = net_dict['pool_type'], net_dict['pool_size'], net_dict['pool_stride']
#     dr, n_neur, opt = net_dict['dr_rate'], net_dict['neur_n'], net_dict['opt']
#     loss, str_metrics = net_dict['loss'], net_dict['str_metrics']
#     prop_sfac, floor_sfac =  net_dict['prop_sfac'], net_dict['prop_sfac']
#     feat_mean, feat_st_dev  = net_dict['feat_mean'], net_dict['feat_st_dev']
#     inputs_n, features_n = net_dict['inputs_n'], net_dict['features_n']
    
#     if weight_init == 'He':
#         init = tf.keras.initializers.HeNormal(seed = 42)
#     if weight_init == 'Gl':
#         init = tf.keras.initializers.GlorotNormal(seed = 42)
    
#     model = keras.Sequential([ layers.Input(shape = (11,)),
#                               EM1DNoise(prop_sfac = prop_sfac, floor_sfac = floor_sfac, feat_mean = feat_mean, feat_st_dev =feat_st_dev,
#                                         name = "EM_Noise_Augmentation")
#                               ] )
    
#     #First set of convolutional layers
#     model.add(keras.layers.Conv1D(filters = k_base_n, kernel_size = (k_size_1, ), strides=1,
#         kernel_initializer = init, padding = "same", input_shape=( inputs_n, features_n), kernel_constraint=max_norm(kc)))
#     model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))    
#     # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
#     model.add(keras.layers.Conv1D(filters=k_base_n, kernel_size=k_size_1, strides=1,
#         kernel_initializer=init, padding="same", kernel_constraint=max_norm(kc)))
#     model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
#     # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
#     model.add(keras.layers.Conv1D(filters=k_base_n, kernel_size=k_size_1, strides=1,
#         kernel_initializer=init, padding="same", kernel_constraint=max_norm(kc)))
#     model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))  
#     # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
#     model.add(keras.layers.Conv1D(filters=k_base_n, kernel_size=k_size_1, strides=1,
#         kernel_initializer=init, padding="same", kernel_constraint=max_norm(kc)))
#     model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))  
#     # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
#     # model.add(keras.layers.Conv1D(filters=k_base_n, kernel_size=k_size_1, strides=1,
#     #     kernel_initializer=init, padding="same", kernel_constraint=max_norm(kc)))
#     # model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))  
#     # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
#     #First set of convolutional layers
#     # keras.layers.Conv1D(filters=k_base_n, kernel_size=k_size_1, strides=1,
#     #     kernel_initializer=init, padding="same", input_shape=( n_inputs,n_features), kernel_constraint=max_norm(kc)),
#     # keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)),    
#     # # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
#     # keras.layers.Conv1D(filters=k_base_n, kernel_size=k_size_1, strides=1,
#     #     kernel_initializer=init, padding="same", kernel_constraint=max_norm(kc)),
#     # keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)),
#     # # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
#     # keras.layers.Conv1D(filters=k_base_n, kernel_size=k_size_1, strides=1,
#     #     kernel_initializer=init, padding="same", kernel_constraint=max_norm(kc)),
#     # keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)),  
#     # # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
#     # keras.layers.Conv1D(filters=k_base_n, kernel_size=k_size_1, strides=1,
#     #     kernel_initializer=init, padding="same", kernel_constraint=max_norm(kc)),
#     # keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init))), 
#     # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
#     # model.add(keras.layers.Conv1D(filters=k_base_n, kernel_size=k_size_1, strides=1,
#     #     kernel_initializer=init, padding="same", kernel_constraint=max_norm(kc)))
#     # model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))  
#     # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
    
   
#     if pool_type == 'avg_pool':
#         model.add(keras.layers.AveragePooling1D(pool_size=n_pool, strides=p_n_str))
#     if pool_type == 'max_pool':
#         model.add( keras.layers.MaxPooling1D(pool_size=n_pool, strides=p_n_str))
     
#     #Second set of convolutional layers
#     model.add(keras.layers.Conv1D(kernel_initializer=init, filters=2*k_base_n, kernel_size=k_size_2, strides=1, padding="same", kernel_constraint=max_norm(kc)))
#     model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
#     # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
#     model.add(keras.layers.Conv1D(kernel_initializer=init, filters=2*k_base_n, kernel_size=k_size_2, strides=1, padding="same",kernel_constraint=max_norm(kc)))
#     model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
#     # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
#     model.add(keras.layers.Conv1D(kernel_initializer=init, filters=2*k_base_n, kernel_size=k_size_2, strides=1, padding="same",kernel_constraint=max_norm(kc)))
#     model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
#     # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
#     model.add(keras.layers.Conv1D(kernel_initializer=init, filters=2*k_base_n, kernel_size=k_size_2, strides=1, padding="same",kernel_constraint=max_norm(kc)))
#     model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
#     # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
#     # model.add(keras.layers.Conv1D(kernel_initializer=init, filters=2*k_base_n, kernel_size=k_size_2, strides=1, padding="same",kernel_constraint=max_norm(kc)))
#     # model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
#     # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
    
    
#     if pool_type == 'avg_pool':
#         model.add(keras.layers.AveragePooling1D(pool_size=n_pool, strides=p_n_str))
#     if pool_type == 'max_pool':
#         model.add( keras.layers.MaxPooling1D(pool_size=n_pool, strides=p_n_str))
    
   
#     #Third set of convolutional layers
#     model.add(keras.layers.Conv1D(kernel_initializer=init, filters=4*k_base_n, kernel_size=k_size_2, strides=1, padding="same",kernel_constraint=max_norm(kc)))
#     model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
#     # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
#     model.add(keras.layers.Conv1D(kernel_initializer=init, filters=4*k_base_n, kernel_size=k_size_2, strides=1, padding="same",kernel_constraint=max_norm(kc)))
#     model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
#     # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
#     model.add(keras.layers.Conv1D(kernel_initializer=init, filters=4*k_base_n, kernel_size=k_size_2, strides=1, padding="same",kernel_constraint=max_norm(kc)))
#     model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
#     # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
#     model.add(keras.layers.Conv1D(kernel_initializer=init, filters=4*k_base_n, kernel_size=k_size_2, strides=1, padding="same",kernel_constraint=max_norm(kc)))
#     model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
#     # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
#     # model.add(keras.layers.Conv1D(kernel_initializer=init, filters=4*k_base_n, kernel_size=k_size_2, strides=1, padding="same",kernel_constraint=max_norm(kc)))
#     # model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
#     # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
    
     
#     if pool_type == 'avg_pool':
#         model.add(keras.layers.AveragePooling1D(pool_size=n_pool, strides=p_n_str))
#     if pool_type == 'max_pool':
#         model.add( keras.layers.MaxPooling1D(pool_size=n_pool, strides=p_n_str))
    
#     # #4th set of convolutional layers
#     # model.add(keras.layers.Conv1D(kernel_initializer=init, filters=8*k_base_n, kernel_size=k_size_2, strides=1, padding="same",kernel_constraint=max_norm(kc)))
#     # model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
#     # # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
#     # model.add(keras.layers.Conv1D(kernel_initializer=init, filters=8*k_base_n, kernel_size=k_size_2, strides=1, padding="same",kernel_constraint=max_norm(kc)))
#     # model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
#     # # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
#     # model.add(keras.layers.Conv1D(kernel_initializer=init, filters=8*k_base_n, kernel_size=k_size_2, strides=1, padding="same",kernel_constraint=max_norm(kc)))
#     # model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
#     # # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
#     # model.add(keras.layers.Conv1D(kernel_initializer=init, filters=8*k_base_n, kernel_size=k_size_2, strides=1, padding="same",kernel_constraint=max_norm(kc)))
#     # model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
#     # # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
#     # model.add(keras.layers.Conv1D(kernel_initializer=init, filters=8*k_base_n, kernel_size=k_size_2, strides=1, padding="same", kernel_constraint=max_norm(kc)))
#     # model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
#     # # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
     
#     # if pool_type == 'avg_pool':
#     #     model.add(keras.layers.AveragePooling1D(pool_size=n_pool, strides=p_n_str))
#     # if pool_type == 'max_pool':
#     #     model.add( keras.layers.MaxPooling1D(pool_size=n_pool, strides=p_n_str))
        
#     model.add(keras.layers.Flatten())
    
#     ### the set of densely connected layers
#     model.add(keras.layers.Dropout(rate = 0.1))
#     # model.add(keras.layers.BatchNormalization())
#     model.add(keras.layers.Dense(units = n_neur))
#     model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
#     model.add(keras.layers.Dropout(rate = dr ))
#     # model.add(keras.layers.BatchNormalization())
#     model.add(keras.layers.Dense(units = n_neur))
#     model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
#     model.add(keras.layers.Dropout(rate = dr ))
#     # model.add(keras.layers.BatchNormalization())
#     model.add(keras.layers.Dense(units = n_neur))
#     model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
#     model.add(keras.layers.Dropout(rate = dr ))
#     # model.add(keras.layers.BatchNormalization())
#     model.add(keras.layers.Dense(units = n_neur))
#     model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
#     # model.add(keras.layers.Dropout(rate = dr))
#     # model.add(keras.layers.BatchNormalization())
#     model.add(keras.layers.Dense(units = 31, kernel_regularizer = tf.keras.regularizers.L2(reg)))
    
#     ###Print Model Architecture
#     print(model.summary())
#     model.compile(optimizer=opt, loss= loss, metrics = str_metrics)
#     return model

def VGG_CNN(w_init, prel_init, k_base_n, k_size_1, cnn_str, k_size_2, kc, dr, n_neur,opt, loss, str_metrics, n_inputs, n_features, n_pool, p_n_str, reg, data_stddev):
    if w_init == 'He':
        init = tf.keras.initializers.HeNormal(seed = 42)
    if w_init == 'Gl':
        init = tf.keras.initializers.GlorotNormal(seed = 42)

    model = keras.Sequential(layers.GaussianNoise(stddev = 0.2))
    
    #First set of convolutional layers
    model.add(keras.layers.Conv1D(filters=k_base_n, kernel_size=k_size_1, strides=cnn_str,
        kernel_initializer=init, padding="same", input_shape=(n_inputs,n_features), kernel_constraint=max_norm(kc)))
    # model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))    
    model.add(keras.layers.Conv1D(filters=k_base_n, kernel_size=k_size_1, strides=cnn_str,
        kernel_initializer=init, padding="same", kernel_constraint=max_norm(kc)))
    # model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))   
    model.add(keras.layers.Conv1D(filters=k_base_n, kernel_size=k_size_1, strides=cnn_str,
        kernel_initializer=init, padding="same", kernel_constraint=max_norm(kc)))
    # model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))  
    # model.add(keras.layers.Conv1D(filters=k_base_n, kernel_size=k_size_1, strides=cnn_str,
    #     kernel_initializer=init, padding="same", kernel_constraint=max_norm(kc)))
    # # model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    model.add(keras.layers.AveragePooling1D(pool_size=n_pool, strides=p_n_str))
    
    #Second set of convolutional layers
    model.add(keras.layers.Conv1D(kernel_initializer=init, filters=2*k_base_n, kernel_size=k_size_2, strides=cnn_str, padding="same", kernel_constraint=max_norm(kc)))
    # model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    model.add(keras.layers.Conv1D(kernel_initializer=init, filters=2*k_base_n, kernel_size=k_size_2, strides=cnn_str, padding="same",kernel_constraint=max_norm(kc)))
    # model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    model.add(keras.layers.Conv1D(kernel_initializer=init, filters=2*k_base_n, kernel_size=k_size_2, strides=cnn_str, padding="same",kernel_constraint=max_norm(kc)))
    # model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    # model.add(keras.layers.Conv1D(kernel_initializer=init, filters=2*k_base_n, kernel_size=k_size_2, strides=cnn_str, padding="same",kernel_constraint=max_norm(kc)))
    # # model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    model.add(keras.layers.AveragePooling1D(pool_size=n_pool, strides=p_n_str)) 
   
    #Third set of convolutional layers
    model.add(keras.layers.Conv1D(kernel_initializer=init, filters=4*k_base_n, kernel_size=k_size_2, strides=cnn_str, padding="same",kernel_constraint=max_norm(kc)))
    # model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    model.add(keras.layers.Conv1D(kernel_initializer=init, filters=4*k_base_n, kernel_size=k_size_2, strides=cnn_str, padding="same",kernel_constraint=max_norm(kc)))
    # model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    model.add(keras.layers.Conv1D(kernel_initializer=init, filters=4*k_base_n, kernel_size=k_size_2, strides=cnn_str, padding="same",kernel_constraint=max_norm(kc)))
    # model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    # model.add(keras.layers.Conv1D(kernel_initializer=init, filters=4*k_base_n, kernel_size=k_size_2, strides=cnn_str, padding="same",kernel_constraint=max_norm(kc)))
    # # model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    model.add(keras.layers.AveragePooling1D(pool_size=n_pool, strides=p_n_str))
    
    #Fourth set of convolutional layers
    model.add(keras.layers.Conv1D(kernel_initializer=init, filters=8*k_base_n, kernel_size=k_size_2, strides=cnn_str, padding="same",kernel_constraint=max_norm(kc)))
    # model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    model.add(keras.layers.Conv1D(kernel_initializer=init, filters=8*k_base_n, kernel_size=k_size_2, strides=cnn_str, padding="same",kernel_constraint=max_norm(kc)))
    # model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    model.add(keras.layers.Conv1D(kernel_initializer=init, filters=8*k_base_n, kernel_size=k_size_2, strides=cnn_str, padding="same",kernel_constraint=max_norm(kc)))
    # model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    # model.add(keras.layers.Conv1D(kernel_initializer=init, filters=8*k_base_n, kernel_size=k_size_2, strides=cnn_str, padding="same",kernel_constraint=max_norm(kc)))
    # # model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    model.add(keras.layers.Flatten())
    
    ### the set of densely connected layers
    model.add(keras.layers.Dense(units = n_neur))
    model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    model.add(keras.layers.Dropout(rate = dr ))
    model.add(keras.layers.Dense(units = n_neur))
    model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    model.add(keras.layers.Dropout(rate = dr ))
    model.add(keras.layers.Dense(units = n_neur))
    model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    # model.add(keras.layers.Dropout(rate = dr ))
    # model.add(keras.layers.Dense(units = n_neur))
    # model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    # model.add(keras.layers.Dropout(rate = dr ))
    # model.add(keras.layers.Dense(units = n_neur))
    # model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    # model.add(keras.layers.Dropout(rate = dr ))
    # model.add(keras.layers.Dense(units = n_neur))
    # model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    # model.add(keras.layers.Dropout(rate = dr ))
    model.add(keras.layers.Dense(units = 31, kernel_regularizer = tf.keras.regularizers.L2(reg)))
    ###Print Model Architecture
    print(model.summary())
    model.compile(optimizer=opt, loss= loss, metrics = str_metrics)
    return model


def cnn_test(w_init, prel_init, k_base_n, k_size_1, cnn_str, k_size_2, kc, dense_kc,
               dr, sp_dr_rate, n_neur,opt, loss, str_metrics, n_inputs, n_features, 
               pool_type, n_pool, p_n_str, reg):
    
    if w_init == 'He':
        init = tf.keras.initializers.HeNormal(seed = 42)
    if w_init == 'Gl':
        init = tf.keras.initializers.GlorotNormal(seed = 42)
    
    model = keras.Sequential()
    
    #First set of convolutional layers
    model.add(keras.layers.Conv1D(filters=k_base_n, kernel_size=k_size_1, strides=1,
        kernel_initializer=init, padding="same", input_shape=( n_inputs,n_features), kernel_constraint=max_norm(kc)))
    model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))    
    model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
    model.add(keras.layers.Conv1D(filters=k_base_n, kernel_size=k_size_1, strides=1,
        kernel_initializer=init, padding="same", kernel_constraint=max_norm(kc)))
    model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
    model.add(keras.layers.Conv1D(filters=k_base_n, kernel_size=k_size_1, strides=1,
        kernel_initializer=init, padding="same", kernel_constraint=max_norm(kc)))
    model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))  
    model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
    model.add(keras.layers.Conv1D(filters=k_base_n, kernel_size=k_size_1, strides=1,
        kernel_initializer=init, padding="same", kernel_constraint=max_norm(kc)))
    model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
    model.add(keras.layers.Conv1D(kernel_initializer=init, filters=k_base_n, kernel_size=k_size_2, strides=1, padding="same", kernel_constraint=max_norm(kc)))
    model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
   
    if pool_type == 'avg_pool':
        model.add(keras.layers.AveragePooling1D(pool_size=n_pool, strides=p_n_str))
    if pool_type == 'max_pool':
        model.add( keras.layers.MaxPooling1D(pool_size=n_pool, strides=p_n_str))
     
    #Second set of convolutional layers
    model.add(keras.layers.Conv1D(kernel_initializer=init, filters=2*k_base_n, kernel_size=k_size_2, strides=1, padding="same", kernel_constraint=max_norm(kc)))
    model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
    model.add(keras.layers.Conv1D(kernel_initializer=init, filters=2*k_base_n, kernel_size=k_size_2, strides=1, padding="same",kernel_constraint=max_norm(kc)))
    model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
    model.add(keras.layers.Conv1D(kernel_initializer=init, filters=2*k_base_n, kernel_size=k_size_2, strides=1, padding="same",kernel_constraint=max_norm(kc)))
    model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
    # model.add(keras.layers.Conv1D(kernel_initializer=init, filters=2*k_base_n, kernel_size=k_size_2, strides=1, padding="same",kernel_constraint=max_norm(kc)))
    # model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
    # model.add(keras.layers.Conv1D(kernel_initializer=init, filters=2*k_base_n, kernel_size=k_size_2, strides=1, padding="same", kernel_constraint=max_norm(kc)))
    # model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
    
    if pool_type == 'avg_pool':
        model.add(keras.layers.AveragePooling1D(pool_size=n_pool, strides=p_n_str))
    if pool_type == 'max_pool':
        model.add( keras.layers.MaxPooling1D(pool_size=n_pool, strides=p_n_str))
    
   
    #Third set of convolutional layers
    model.add(keras.layers.Conv1D(kernel_initializer=init, filters=4*k_base_n, kernel_size=k_size_2, strides=1, padding="same",kernel_constraint=max_norm(kc)))
    model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
    model.add(keras.layers.Conv1D(kernel_initializer=init, filters=4*k_base_n, kernel_size=k_size_2, strides=1, padding="same",kernel_constraint=max_norm(kc)))
    model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
    model.add(keras.layers.Conv1D(kernel_initializer=init, filters=4*k_base_n, kernel_size=k_size_2, strides=1, padding="same",kernel_constraint=max_norm(kc)))
    model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
    # model.add(keras.layers.Conv1D(kernel_initializer=init, filters=4*k_base_n, kernel_size=k_size_2, strides=1, padding="same",kernel_constraint=max_norm(kc)))
    # model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
    # model.add(keras.layers.Conv1D(kernel_initializer=init, filters=4*k_base_n, kernel_size=k_size_2, strides=1, padding="same", kernel_constraint=max_norm(kc)))
    # model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    # model.add(keras.layers.SpatialDropout1D(rate = sp_dr_rate))
    
    if pool_type == 'avg_pool':
        model.add(keras.layers.AveragePooling1D(pool_size=n_pool, strides=p_n_str))
    if pool_type == 'max_pool':
        model.add( keras.layers.MaxPooling1D(pool_size=n_pool, strides=p_n_str))
    
    
    model.add(keras.layers.Flatten())
    
    ### the set of densely connected layers
    model.add(keras.layers.Dropout(rate = dr))
    model.add(keras.layers.Dense(units = n_neur, kernel_constraint =max_norm(dense_kc)))
    model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    model.add(keras.layers.Dropout(rate = dr ))
    model.add(keras.layers.Dense(units = n_neur, kernel_constraint =max_norm(dense_kc)))
    model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    model.add(keras.layers.Dropout(rate = dr ))
    model.add(keras.layers.Dense(units = n_neur, kernel_constraint =max_norm(dense_kc)))
    model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    model.add(keras.layers.Dropout(rate = dr ))
    model.add(keras.layers.Dense(units = n_neur, kernel_constraint=max_norm(dense_kc)))
    model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
    model.add(keras.layers.Dropout(rate = dr))
    model.add(keras.layers.Dense(units = n_neur, kernel_constraint =max_norm(dense_kc)))
    model.add(keras.layers.PReLU(alpha_initializer = tf.keras.initializers.constant(prel_init)))
   
    model.add(keras.layers.Dense(units = 30, kernel_regularizer = tf.keras.regularizers.L2(reg)))
    
    ###Print Model Architecture
    print(model.summary())
    model.compile(optimizer=opt, loss= loss, metrics = str_metrics)
    return model


