 # -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 14:28:36 2019

@author: WYU54
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import os
import joblib
from sympy import symbols, diff
from matplotlib.widgets import Slider, Button
import matplotlib.ticker as mticker
import matplotlib

#%%
 

def preprocessdf(filepath = '24V1.00Pa.csv', mass = False):
    """
    Preprocess the trajectory data in the given CSV file.

    Parameters:
    filepath (str): The path to the CSV file.
    mass (bool): A flag indicating whether or not to include mass in the preprocessed data.

    Returns:
    pd.DataFrame: The preprocessed trajectory data.
    """

    # Load the data from the CSV file.
    df = pd.read_csv(filepath)

    # Center the x, y, and z coordinates around the mean.
    df.x -= df.x.mean()
    df.y -= df.y.mean()
    try:
        df.z -= df.z.mean()
    except:
        # Pass if z column does not exist
        pass

    try:
        # Create a list of columns to be used in the dataframe.
        cols = ['frame','particle','xmm','ymm','zmm']
        if mass:
            # If mass is true, include the mass column in the data.
            cols += ['mass']
        # Select the desired columns from the dataframe.
        df = df[cols]
        # Rename 'xmm', 'ymm', 'zmm' columns to 'x', 'y', 'z'
        df.rename(columns = {'xmm':'x','ymm':'y','zmm':'z'},inplace = True)
    except:
        # Print an error message if 'zmm' column is not found in the dataframe.
        print('cannot find zmm')
        pass

    # Set 'frame' and 'particle' as multi-index and drop unneeded columns ('Unnamed: 0', 'temp1'/'mass' based on condition, 'time_s').
    df = df.set_index(['frame','particle'], drop = True).drop(columns = ['Unnamed: 0','tish' if mass else 'mass','time_s'],errors = 'ignore').unstack()

    # Return the preprocessed dataframe.
    return df


# Define the symbol t for sympy expression
t = symbols('t')

# Define the function w which is a sympy expression involving t
w = (t**2-1)**2

def w_to_coef(w,tau = 6):
    """
    Function to calculate the coefficients of w using Simpson's rule.

    Parameters:
    w (sympy expression): The function to integrate.
    tau (int): The number of divisions in the range -1 to 1.

    Returns:
    tf.Tensor: The coefficients of w.
    """

    # Define the points t at which to evaluate w
    tbars = np.linspace(-1,1,tau+1)
    
    # Evaluate w at the points tbars
    ws = np.array([w.subs('t',i) for i in tbars])
    
    # Define the weights for Simpson's rule
    simpson_weights = np.array([1,4] + [2,4] * (tau//2-1) +[1]) / 3
    
    # Handle edge cases where the first and/or last value of ws is 0
    begin = 0 if ws[0] != 0 else 1
    end = tau + 1 if ws[-1] !=0 else tau
    
    # Apply the weights to the function values ws and convert to float64
    coef = (simpson_weights * ws).astype(np.float64)
    
    # Convert the numpy array to a tensorflow constant and return
    return tf.constant(coef[begin:end],dtype = tf.float32)

    

#%%
'''
TensorFlow
'''
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, LeakyReLU,Dropout
from tensorflow.python.keras.engine.base_layer import Layer
from sklearn.model_selection import train_test_split
# since the model is small, it's better to run on cpu
tf.config.set_visible_devices([], 'GPU')   
def get_XY(X, w, tau, deltat, ndim = 3, nparticles = 9, whichparticle = 0, gamma = 0.95, n_out = 2):
    """
    Preprocesses the data for a tensorflow model. 
    
    Parameters:
    X (np.array): Input data.
    w (sympy expression): The function to integrate.
    tau (int): The number of divisions in the range -1 to 1.
    deltat (float): Time step size.
    ndim (int): Number of dimensions. Default is 3.
    nparticles (int): Number of particles. Default is 9.
    whichparticle (int): Index of the particle of interest. Default is 0.
    gamma (float): Discount factor. Default is 0.95.
    n_out (int): Number of output dimensions. Default is 2.
    
    Returns:
    tf.Tensor: The preprocessed input data.
    tf.Tensor: The preprocessed output data for first derivative.
    tf.Tensor: The preprocessed output data for second derivative.
    """

    # Re-order the columns of X.
    colidx = list(range(nparticles * ndim))
    for temp1 in range(ndim):
        colidx[temp1] = temp1 + whichparticle * ndim
        colidx[temp1 + whichparticle * ndim] = temp1
    X = X[..., colidx]
    
    # Convert X to a 3D array.
    X = tf.cast(tf.stack([X[i:i-tau-1] for i in range(tau+1)], axis = 1), tf.float32)

    # Compute coefficients for w and its derivatives using Simpson's rule.
    w_coef = w_to_coef(w, tau)
    wdot_coef = w_to_coef(w.diff('t'), tau)
    wdotdot_coef = w_to_coef(w.diff('t', 't'), tau)

    # Check for correct dimensions of w's derivatives coefficients.
    if len(wdotdot_coef) != tau + 1 or len(wdot_coef) != tau - 1:
        raise ValueError('Incorrect dimensions for w coefficients!')
        
    # Compute the product of X and the coefficients of w's derivatives.
    wdotdotr = tf.einsum('ijk,j->ik', X[:, :, :n_out], wdotdot_coef) / (deltat * tau / 2) ** 2
    wdotr = tf.einsum('ijk,j->ik', X[:, 1:-1, :n_out], wdot_coef) / (tau * deltat / 2)

    # Return the preprocessed input and output data.
    return X[:, 1:-1, :], wdotr, wdotdotr

def get_all_XY(df, w, tau, deltat, ndim = 3, nparticles = 9, gamma = 0.95, n_out = 2, **kw):
    """
    Preprocesses the data for all particles for a tensorflow model.
    
    Parameters:
    df (pd.DataFrame): Input data.
    w (sympy expression): The function to integrate.
    tau (int): The number of divisions in the range -1 to 1.
    deltat (float): Time step size.
    ndim (int): Number of dimensions. Default is 3.
    nparticles (int): Number of particles. Default is 9.
    gamma (float): Discount factor. Default is 0.95.
    n_out (int): Number of output dimensions. Default is 2.
    
    Returns:
    tf.Tensor: The preprocessed input data for all particles.
    tf.Tensor: The preprocessed output data for first derivative for all particles.
    tf.Tensor: The preprocessed output data for second derivative for all particles.
    """
    
    # Initialize lists to store the preprocessed data for each particle.
    Xlist = []
    Ylist = []
    Vlist = []
    
    # Loop over all particles.
    for i in range(nparticles):
        # Preprocess the data for the i-th particle.
        X, V, Y = get_XY(df, w, tau, deltat, ndim, nparticles, i, gamma, n_out)
        # Append the preprocessed data to the respective lists.
        Xlist.append(X)
        Ylist.append(Y)
        Vlist.append(V)
    
    # Concatenate the preprocessed data for all particles.
    return tf.concat(Xlist, axis = 0), tf.concat(Vlist, axis = 0), tf.concat(Ylist, axis = 0)

# Alias for get_all_XY
get_XY_all = get_all_XY


def microprocess(df,dz = 2.5):
        ex = 30
        temp1 = df.stack().reset_index()
        temp1.frame = temp1.frame*ex + round(temp1.z/dz*ex).astype(int)
        frame = temp1.frame
        temp2 = temp1.set_index(['frame','particle']).unstack().reindex(range(frame.min(),frame.max()+1)).interpolate()
        temp1 = temp2.loc[list(range(ex,frame.max()-ex,ex))].stack().reset_index()
        temp1.frame = temp1.frame // (ex)
        ##################
        return temp1.set_index(['frame','particle']).unstack()
    
    
def get_data(fname, val_begin=[0.3, 0.7], *a, **kw):
    '''
    Parameters
    ----------
    fname : str, optional
        File path of the CSV data. The default is r'D:\Wentao\2022_videos\0326\17.3V1.00Pa_9p.csv'.
    val_begin : list of 0-1 floats
        The beginning of the sections used for testing data. The total test data length will always be 10% [0.3, 0.7].
    *a : tuple
        Additional parameters for get_all_XY.
    **kw : dict
        Additional keyword parameters.
    '''
    gap = 1
    df = preprocessdf(fname, mass=1)
    if '15p' in fname:
        # Only for this specific data
        df = microprocess(df)
    df = df.stack().reset_index()
    df['z'] -= df['z'].mean()
    df = df.set_index(['frame', 'particle']).unstack()
    if '24V' in fname:
        # Only for this specific data
        df = df.iloc[:-555]
    if 'ndim' in kw and kw['ndim'] == 4:
        #calculate s_i, with specified method and scale, if exist in **kw
        temp1 = df.stack().reset_index()
        temp1['z'] -= temp1['z'].mean()
        temp1['x'] -= temp1['x'].mean()
        temp1['y'] -= temp1['y'].mean()
        nparticles = kw['nparticles']
        method = kw.get('method', 'mean')
        scale = kw.get('scale', 0.1 if nparticles < 10 else 0.2)
        if callable(method):
            descriptors = [method(temp1.loc[temp1['particle'] == p, 'z'].values) * scale for p in range(nparticles)]
        elif method == 'std':
            descriptors = temp1.groupby('particle')['z'].std() * scale
        elif method == 'mean':
            descriptors = temp1.groupby('particle')['z'].mean() * scale
        else:
            raise ValueError('Method must be "std", "mean", or callable.')
        descriptors -= descriptors.mean()
        temp1['zm'] = descriptors[temp1['particle']].values
        temp1 = temp1.iloc[nparticles:]
        df = temp1.drop(columns='mass', errors='ignore').set_index(['frame', 'particle']).unstack()
        #print(df.head(5))
    df = df.swaplevel(axis = 1)
    df = df.reindex(sorted(df.columns), axis=1)
    #separate train test data by the percentiles given at val_begin 
    l = df.values.shape[0]
    begins = [int(l*v) for v in val_begin] + [l+1]
    dur = int(0.1*l/(len(begins)-1))
    ends = [0] + [begin + dur for begin in begins]
    X_trains = []
    V_trains = []
    Y_trains = []
    for i in range(len(begins)):
        X, V, Y = get_XY_all(df.values[ends[i]:begins[i]],*a,**kw)
        X = X[::gap]
        V = V[::gap]
        Y = Y[::gap]
        X_trains.append(X)
        V_trains.append(V)
        Y_trains.append(Y)
    X_trains = tf.concat(X_trains,axis = 0)
    Y_trains = tf.concat(Y_trains,axis = 0)
    V_trains = tf.concat(V_trains,axis = 0)
    X_tests = []
    V_tests = []
    Y_tests = []
    for i in range(len(begins)-1):
        X, V, Y = get_XY_all(df.values[begins[i]:ends[i+1]],*a,**kw)
        X = X[::gap]
        V = V[::gap]
        Y = Y[::gap]
        X_tests.append(X)
        V_tests.append(V)
        Y_tests.append(Y)
    X_tests = tf.concat(X_tests,axis = 0)
    Y_tests = tf.concat(Y_tests,axis = 0)
    V_tests = tf.concat(V_tests,axis = 0)
    # X, V, Y are 3D tensor, 2D tensor, 2D tensor, as specified in the paper data process section
    return [X_trains,V_trains], [X_tests, V_tests], Y_trains, Y_tests

# basic dense neural net
class MyDense(keras.Model):
    def __init__(self,n_layers = 3,n_neurons = 10, n_out = 2, **kw):
        super().__init__(**kw)
        self.hidden = [keras.layers.Dense(n_neurons,'tanh' if i==1 else 'elu',kernel_initializer ='he_normal', kernel_regularizer = keras.regularizers.l2(.01)) for i in range(n_layers)]
        self.out_layer = keras.layers.Dense(n_out)
    def call(self,inputs):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
        return self.out_layer(Z)

def dense_sequential(n_layers = 3, n_neurons = 32, n_out = 2):
    mdl = Sequential()
    for i in range(n_layers):
        mdl.add(Dense(n_neurons, kernel_initializer ='he_normal', kernel_regularizer = keras.regularizers.l2(.01)))
        if i != 1:
            mdl.add(LeakyReLU(alpha = 0.1))
        else:
            mdl.add(Activation('tanh'))
    mdl.add(Dense(n_out))
    return mdl

    
class Myint(keras.Model):
    def __init__(self,n_layers = 3,n_neurons = 10, n_out = 2,n_in = 3, **kw):
        super().__init__(**kw)
        # Initialize a sequential model using the dense_sequential function defined earlier
        self.mdl = dense_sequential(n_layers, n_neurons, n_out-1)
        self.n_out = n_out
        self.n_in = n_in
    
    @tf.function
    def call(self,inputs):
        #inputs are: (xself, yself, zself,sself, xother, yother,zother, sother)
        # outputs are: fij_x*rij, fij_y*rij
        # Computing the squared separation in xy plane and z direction between two particles
        sep_xy2 = tf.math.square(inputs[...,self.n_in+1] - inputs[...,1]) + tf.math.square(inputs[...,self.n_in] - inputs[...,0])
        sep_z2 = tf.math.square(inputs[...,self.n_in+2] - inputs[...,2])

        # Total squared separation is sum of squared separation in xy plane and z direction
        sep_r2 = sep_xy2 + sep_z2

        # Compute the inverse square root of the total separation
        sep_r_2 = 1 / sep_r2

        # Compute the direction of interaction in xy plane
        dir_xy = inputs[...,self.n_in:self.n_in+2] - inputs[...,0:2]
        dir_xy = tf.einsum('ijk,ij->ijk',dir_xy,sep_r_2)
        #dir_xy = (delta x/ delta r^2, delta y / delta r^2)

        # Prepare the input tensor by stacking the z coordinates and squared separation in xy plane
        Z = tf.stack([inputs[...,2],inputs[...,self.n_in+2],sep_xy2],axis = -1)

        # If there are additional inputs, concatenate them
        if self.n_in > 3:
            Z = tf.concat([Z,inputs[...,3:self.n_in],inputs[...,3+self.n_in:]],axis = -1)
        
        # Compute the amplitude of the force of interaction using the sequential model
        F_amp = self.compute_amp(Z)

        # Depending on the number of outputs, return force vector in the xy plane or the 3D force vector
        if self.n_out == 2:
            return tf.einsum('...,...k->...k',F_amp, dir_xy)
        elif self.n_out == 3:
            Fxy = tf.einsum('...,...k->...k',F_amp[...,0], dir_xy)
            F = tf.concat([Fxy,F_amp[...,1:]],axis = -1)
            return F
        else:
            return F_amp
        
    @tf.function
    def compute_amp(self,inputs3d):
        '''
        Note that the output is interaction * rho
        '''
        # Pass the inputs through the sequential model to compute the force amplitude
        Z = self.mdl(inputs3d)

        # Depending on the number of outputs, return the computed force amplitude
        if self.n_out == 2:
            return tf.squeeze(Z)
        else:
            return Z


 
class MyModel1(keras.Model):
    def __init__(self, weight_vector, n_particles, n_dim, n_out = 2,int_neurons = 32, conf_neurons = 16, n_layers = 3, **kw):
        super().__init__(**kw)

        self.n_particles = n_particles  # Number of particles in the system
        self.n_dim = n_dim  # Number of dimensions
        self.n_out = n_out  # Number of output features

        # Interaction model between particle pairs
        self.F_int = Myint(n_layers,int_neurons,n_out,n_in = n_dim)

        # Dense model for environmental force
        self.F_conf = MyDense(n_layers,conf_neurons,n_out)

        # Weight vector for combining force contributions(convolution)
        self.weight_vector = tf.cast(weight_vector,tf.float32)

        # Dense model for damping coefficient
        self.g_w = MyDense(2,16,1)

        # Parameter for extra loss
        self.tau = len(self.weight_vector) + 1

        # Tensor for extra loss calculation
        self.loss_X = tf.constant([[z1,z2,r] for z1 in np.arange(-0.5,0.5,0.02) for z2 in np.arange(-0.5,0.5,0.02) for r in np.arange(1.4,16,0.1)**2],tf.float32)


    @tf.function
    def get_extra_loss(self):
        # Compute extra loss to ensure that the interaction force approaches zero for large particle separation
        X = tf.pow(tf.random.uniform([32,3],[-0.8,-0.8,1.7],[.8,.8,2.5]),3)
        if self.n_out == 2:
            return tf.keras.regularizers.l2(100)(self.F_int.compute_amp(X))

    #@tf.function
    def call(self,X):
        X,wdotr = X
        s = X[:,0,self.n_dim -1]

        # Compute damping coefficient from input
        gamma = self.g_w(s[...,tf.newaxis])[...,0]

        # Compute environmental force
        conf_force = self.F_conf(X[...,:self.n_dim])

        # Apply weight vector to environmental force
        w_conv_conf_force = tf.einsum('ijk,j->ik',conf_force,self.weight_vector)

        # Initialize list of forces
        force_list = [w_conv_conf_force]

        # Compute interaction forces and apply weight vector
        for i in range(self.n_particles-1):  
            slices = list(range(self.n_dim)) + list(range(self.n_dim*(i+1), self.n_dim*(i+2)))
            int_force = self.F_int(tf.gather(X,indices = slices,axis = -1))
            w_conv_int_force = tf.einsum('ijk,j->ik',int_force,self.weight_vector)
            force_list.append(w_conv_int_force)

        # Stack the forces
        force_list = tf.stack(force_list)

        # Return the sum of forces with added damping force
        return tf.reduce_sum(force_list,axis = 0) * 100 + tf.einsum('i,ij->ij',gamma,wdotr) 


def train(mdl,X1,X2,Y1,Y2,e1 = 25):
    # Initialize mini-batch size
    mb = 64

    # If the number of epochs is greater than 15, start with a higher learning rate
    if e1>15:
        mdl.compile(loss = 'mse', metrics = [myr2()], optimizer = tf.keras.optimizers.RMSprop(lr = 0.005, momentum = 0.01))
        h = mdl.fit(X1,Y1,mb,5,validation_data = (X2,Y2))

    # Compile and train the model with default learning rate
    mdl.compile(loss = 'mse', metrics = [myr2()])
    h = mdl.fit(X1,Y1,mb,e1,validation_data = (X2,Y2))

    # Gradually reduce the learning rate and continue training
    mdl.compile(loss = 'mse', metrics = [myr2()],optimizer = tf.keras.optimizers.RMSprop(lr = 0.0003, momentum = 0.01))
    print('Change lr to 0.0003')
    h = mdl.fit(X1,Y1,mb,round(e1),validation_data = (X2,Y2))

    mdl.compile(loss = 'mse', metrics = [myr2()],optimizer = tf.keras.optimizers.RMSprop(lr = 0.0001, momentum = 0.01))
    print('Change lr to 0.0001')
    h = mdl.fit(X1,Y1,mb,round(e1),validation_data = (X2,Y2))

    mdl.compile(loss = 'mse', metrics = [myr2()],optimizer = tf.keras.optimizers.RMSprop(lr = 0.00003, momentum = 0.01))
    print('Change lr to 0.00003')
    h = mdl.fit(X1,Y1,mb,round(e1*.7),validation_data = (X2,Y2))

    mdl.compile(loss = 'mse', metrics = [myr2()],optimizer = tf.keras.optimizers.RMSprop(lr = 0.00001, momentum = 0.01))
    print('Change lr to 0.00001')
    h = mdl.fit(X1,Y1,mb,round(e1*0.4),validation_data = (X2,Y2))

    return h

def decay_train_huber(mdl,X1,X2,Y1,Y2,scale = 20, delta = 1000, nepochs = None):
    # Initialize mini-batch size based on scale
    if nepochs is not None:
        scale = nepochs//20 
    if scale < 11:
        mb = 128
    else:
        mb = 64

    # Set validation data
    if X2 is None:
        val = None
    else:
        val = (X2,Y2)

    # Compile the model with Huber loss and exponentially decaying learning rate
    mdl.compile(loss = tf.keras.losses.Huber(delta = delta), metrics = [myr2(),tf.keras.losses.MeanSquaredError(name = 'M')], optimizer = tf.keras.optimizers.RMSprop(lr = 0.004))

    # Fit the model
    h = mdl.fit(X1,Y1,mb,scale * 20 + 10,validation_data = val,callbacks = [tf.keras.callbacks.LearningRateScheduler(exp_decay_epoch(scale),0)])

    return h

def exp_decay_epoch(scale = 20):
    def decay(epoch,lr):
        if epoch <= scale:
            return 0.004 * 0.25 **(epoch/scale)
        elif scale < epoch < scale*15:
            return 0.0013 * 0.1 ** (epoch/(scale*14))
        elif scale*15 <= epoch < scale*20:
            return 0.0001 * 0.1 **((epoch-scale*15)/(scale*5))
        else:
            return lr * 0.9
    return decay
    
class myr2(tf.keras.metrics.Metric):
    """
    This class implements the r-squared (R2) metric also known as 
    the coefficient of determination, which is a statistical measure
    that represents the proportion of the variance for a dependent variable 
    that's explained by an independent variable or variables in a regression model.
    """

    def __init__(self, name='r2', **kwargs):
        """
        Initialize the metric with its name and any other parameters. 
        Also initializes the mean squared error (mse) and variance (var) as variables with initial value zero.
        """
        super().__init__(name=name, **kwargs)
        self.mse = self.add_weight(name='mse', initializer='zeros')
        self.var = self.add_weight(name='var', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        This method updates the state of the metric with new data: true values, predicted values and sample weights.
        It calculates the new values of mean squared error and variance, updates the variables.
        """
        # Compute Mean Squared Error between true and predicted values
        mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
        # Compute variance of true values
        var = tf.math.reduce_mean(y_true**2)

        mse = tf.cast(mse, self.dtype)
        var = tf.cast(var, self.dtype)
        
        # If there is a sample weight, apply it to the mse and var
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, mse.shape)
            mse = tf.multiply(mse, sample_weight)
            var = tf.multiply(var, sample_weight)
        
        # Update the mse and var variables
        self.mse.assign_add(tf.reduce_sum(mse))
        self.var.assign_add(tf.reduce_sum(var))

    def result(self):
        """
        This method returns the final result of the metric after all calculations,
        which is 1 minus the ratio of the mse and var.
        """
        return 1-self.mse/self.var

def cross_val_dat(fname, w, tau, nparticles=9, ndim=4, is64=False, **kw):
    """
    Perform cross-validation on the dataset to evaluate the model performance.
    It creates a list of models and their corresponding history objects 
    after training on different splits of the data.

    Parameters:
    - fname: Name of the file containing the data
    - w: The weight vector
    - tau: Parameter in the weight vector
    - nparticles: Number of particles
    - ndim: Number of dimensions
    - is64: A flag indicating whether to use 64 internal neurons or not.
    - **kw: Additional keyword arguments

    Returns:
    - hlist: A list of history objects returned by model training
    - mdllist: A list of trained models
    """
    hlist = []  # list to store history objects
    mdllist = []  # list to store trained models

    for i in range(10):  # loop for cross validation
        print('looop', i)
        # Split the data
        X1, X2, Y1, Y2 = get_data(fname, [i * 0.1], w, tau, nparticles=nparticles, ndim=ndim, **kw)
        s = X2[0][0, 3, 3::4]

        # Define the model
        if is64:
            mdl = MyModel1(w_to_coef(w, tau), nparticles, ndim, 2, 64, 32, add_loss=False)
        else:
            mdl = MyModel1(w_to_coef(w, tau), nparticles, ndim, 2, 32, 16, add_loss=False)

        # Determine scale and delta based on the filename
        if '9p' in fname:
            scale = 20
            delta = 80
        else:
            scale = 13 + 5 * ('10p' in fname or '0.75' in fname)
            if '11p' in fname:
                scale = 25
            delta = (Y2.numpy().std() * 0.1 + 0.9 * Y1.numpy().std()) * 0.25
            print(delta)

        # Perform training using Huber loss
        h = decay_train_huber(mdl, X1, X2, Y1, Y2, scale, delta)

        # Append the model and its training history to their respective lists
        hlist.append(h)
        mdllist.append(mdl)

    # Return the lists of models and their training histories
    return hlist, mdllist

def get_interaction(mdl, z1, z2, rho, s1, s2):
    '''
    Compute the interaction between a pair of particles.
    
    Parameters
    ----------
    mdl: keras.Model
        The interaction model object from class MyModel1.
    z1, z2, rho, s1, s2: float or list of float
        Parameters that define the state of the particles. Can be either numbers
        or lists of numbers of the same length.
        
    Returns
    -------
    numpy.ndarray
        The interaction computed by the model, divided by the provided rho values.
        
    '''
    # Attempt to get the length of z1. If this fails, it means z1 is not iterable
    # (i.e., it's a single number), and we convert z1, z2, rho, s1, s2 to lists.
    try:
        len(z1)
    except:
        z1 = [z1]
        z2 = [z2]
        rho = [rho]
        s1 = [s1]
        s2 = [s2]
    
    # Convert the parameters to a NumPy array and transpose it
    X = np.array([z1, z2, rho, s1, s2]).T

    # Compute the interaction using the model and divide it by rho
    return mdl.F_int.compute_amp(X).numpy() / -np.array(rho)


def get_confinement(mdl, x, y, z, s):
    '''
    Compute the confinement force on a particle.
    
    Parameters
    ----------
    mdl: keras.Model
        The interaction model object from class MyModel1.
    x, y, z, s: float or list of float
        Parameters that define the state of the particle. Can be either numbers
        or lists of numbers of the same length.
        
    Returns
    -------
    numpy.ndarray
        The confinement force computed by the model.
    '''
    # Attempt to get the length of x. If this fails, it means x is not iterable
    # (i.e., it's a single number), and we convert x, y, z, s to lists.
    try:
        len(x)
    except TypeError:
        x = [x]
        y = [y]
        z = [z]
        s = [s]

    # Convert the parameters to a NumPy array and transpose it
    X = np.array([x, y, z, s]).T

    # Compute the confinement force using the model
    return mdl.F_conf(X).numpy()

def get_gamma(mdl, s):
    '''
    Compute the damping force coefficient.

    Parameters
    ----------
    mdl: keras.Model
        The interaction model object from class MyModel1.
    s: float or list of float
        Parameter that defines the state of the particle. Can be either a number
        or a list of numbers.

    Returns
    -------
    numpy.ndarray
        The damping force coefficient computed by the model.
    '''
    # Attempt to get the length of s. If this fails, it means s is not iterable
    # (i.e., it's a single number), and we convert s to a list.
    try:
        len(s)
    except TypeError:
        s = [s]

    # Convert the parameter to a NumPy array and expand its dimensions
    S = np.array(s)[..., np.newaxis]

    # Compute the damping force coefficient using the model
    return mdl.g_w(S).numpy()[..., 0]

