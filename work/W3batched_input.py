#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:32:47 2024

@author: vebjorntandberg
"""

import autograd.numpy as np  # We need to use this numpy wrapper to make automatic differentiation work later
from autograd import grad, elementwise_grad
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def create_layers(network_input_size, layer_output_sizes):
    layers = []

    i_size = network_input_size
    for layer_output_size in layer_output_sizes:
        W = np.random.randn(layer_output_size, i_size)
        b = np.random.randn(layer_output_size)
        layers.append((W, b))

        i_size = layer_output_size
    return layers


def feed_forward(input_var, layers, activation_funcs):
    a = input_var
    for (W, b), activation_func in zip(layers, activation_funcs):
        z = W @ a + b
        a = activation_func(z)
    return a


def cost(layers, input_var, activation_funcs, target):
    predict = feed_forward(input_var, layers, activation_funcs)
    return mse(predict, target)

# Defining some activation functions
def ReLU(z):
    return np.where(z > 0, z, 0)


# Derivative of the ReLU function
def ReLU_der(z):
    return np.where(z > 0, 1, 0)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def mse(predict, target):
    return np.mean((predict - target) ** 2)


def mse_der(predict, target):
    n = max(predict.shape)
    return 2*(predict - target)*(1/n)

def sigmoid_der(z):
    return sigmoid(z)*(1-sigmoid(z))


def feed_forward_saver(input_var, layers, activation_funcs):
    layer_inputs = []
    zs = []
    a = input_var
    for (W, b), activation_func in zip(layers, activation_funcs):
        layer_inputs.append(a)
        z = W @ a + b
        a = activation_func(z)

        zs.append(z)

    return layer_inputs, zs, a

def backpropagation(input_var, layers, activation_funcs, target, activation_ders, cost_der=mse_der):
    layer_inputs, zs, predict = feed_forward_saver(input_var, layers, activation_funcs)

    layer_grads = [() for layer in layers]

    # We loop over the layers, from the last to the first
    dC_dz = None

    for i in reversed(range(len(layers))):
        layer_input, z, activation_der = layer_inputs[i], zs[i], activation_ders[i]
        
        if i == len(layers) - 1:
            # For last layer we use cost derivative as dC_da(L) can be computed directly
            dC_da = cost_der(predict,target)
        else:
            # For other layers we build on previous z derivative, as dC_da(i) = dC_dz(i+1) * dz(i+1)_da(i)
            (W, b) = layers[i + 1]
            dC_da =  dC_dz @ W
        da_dz = activation_der(z)
        dC_dz = dC_da*da_dz
        
        dC_db = dC_da*da_dz
        
        dz_dW = np.tile(layer_input, (max(dC_db.shape), 1))
        dC_dW = np.diag(dC_da * da_dz) @ dz_dW
        print(dC_dW.shape)
        
        layer_grads[i] = (dC_dW, dC_db)

    return layer_grads


##########################################################################
######## Functionality For Batched inputs ################################
##########################################################################


def create_layers_batch(network_input_size, layer_output_sizes):
    layers = []

    i_size = network_input_size
    for layer_output_size in layer_output_sizes:
        W = np.random.randn(i_size, layer_output_size)
        b = np.random.randn(layer_output_size)
        layers.append((W, b))

        i_size = layer_output_size
    return layers


def feed_forward_batch(inputs, layers, activation_funcs):
    a = inputs
    for (W, b), activation_func in zip(layers, activation_funcs):
        z = a @ W  + b
        a = activation_func(z)

    return a

def feed_forward_saver_batch(input_var, layers, activation_funcs):
    layer_inputs = []
    zs = []
    a = input_var
    for (W, b), activation_func in zip(layers, activation_funcs):
        layer_inputs.append(a)
        z =  a @ W + b
        a = activation_func(z)

        zs.append(z)

    return layer_inputs, zs, a

def backpropagation_batch(input_var, layers,
                          activation_funcs, target,
                          activation_ders,
                          batch_size=5,
                          cost_der=mse_der):
    
    batch_size = target.shape[0]
    
    layer_inputs, zs, predict = feed_forward_saver_batch(input_var, layers, activation_funcs)

    layer_grads = [() for layer in layers]

    # We loop over the layers, from the last to the first
    dC_dz = None

    for i in reversed(range(len(layers))):
        layer_input, z, activation_der = layer_inputs[i], zs[i], activation_ders[i]
        
        if i == len(layers) - 1:
            # For last layer we use cost derivative as dC_da(L) can be computed directly
            dC_da = cost_der(predict,target)
        else:
            # For other layers we build on previous z derivative, as dC_da(i) = dC_dz(i+1) * dz(i+1)_da(i)
            (W, b) = layers[i + 1]
            dC_da =   dC_dz @ (W.T)  
            #dC_da =  dC_dz @ W
            
            
        da_dz = activation_der(z)
        dC_dz = dC_da*da_dz
        
        dC_db = (dC_da*da_dz).mean(axis=0)
        
        diag_dC_dz = np.diag(dC_dz.ravel())
        
        num_repeats = max(dC_db.shape)
        dz_dW = np.repeat(layer_input, num_repeats, axis=0)
        
        dC_dW = diag_dC_dz @ dz_dW
        
        # Reshape the matrix to k x m x n
        n = int(dC_dW.shape[0]/batch_size)
        m = int(dC_dW.shape[1])
        
        tensor_dC_dW = dC_dW.reshape((batch_size, n, m))

        # Calculate the mean along the first axis (axis=0)
        average_dC_dW = np.mean(tensor_dC_dW, axis=0)

        print(f'batch shapes is {average_dC_dW.shape}')
        
        layer_grads[i] = (average_dC_dW, dC_db)

    return layer_grads


if __name__ == '__main__':
    np.random.seed(1)

    network_input_size = 2
    layer_output_sizes = [3,4]
    activation_funcs = [sigmoid, ReLU]
    activation_ders = [sigmoid_der, ReLU_der]
    
    layers = create_layers(network_input_size, layer_output_sizes)
    layers_batch = create_layers_batch(network_input_size, layer_output_sizes)
    
    
    x = np.arange(1,5,1).reshape(2, network_input_size)  
    
    layers_saved = feed_forward_saver_batch(x, layers_batch, activation_funcs)
    output = feed_forward_batch(x, layers_batch, activation_funcs)
    
    target = np.random.rand(2, 4) #Remember to change this to 4

    
    layer_grads = backpropagation_batch(x, layers_batch, activation_funcs, target, activation_ders)
    
    
        
    
    x = np.random.rand(network_input_size)
    target = np.random.rand(4) #Remember to change this to 4
    
    layer_grads = backpropagation(x, layers, activation_funcs, target, activation_ders)
    
    cost_grad = grad(cost, 0)
    cost_grad(layers, x, activation_funcs, target)
    
    
    
    
        