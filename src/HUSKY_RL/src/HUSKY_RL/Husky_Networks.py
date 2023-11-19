# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 11:19:48 2023

@author: Breach
"""

import torch
import itertools
import numpy as np
import torch.nn as nn


class CustomNetwork(nn.Module):
    def __init__(self, layer_params,
                 default,
                 input_dim,
                 ouput_dim,
                 net_type,
                 output_activation):
        
        super(CustomNetwork, self).__init__()
        if default:
           layer_params =self._default_nets(input_dim,ouput_dim,net_type,output_activation)
        self.layers = nn.ModuleList()
        for layer_desc in layer_params:
            layer = self.get_layer(layer_desc)
            self.layers.append(layer)
            if layer_desc[-1] is not None:
                activation_fn = self.get_activation(layer_desc[-1])
                self.layers.append(activation_fn)
                
    def _default_nets(self,input_dim,
                      ouput_dim,
                      net_type,
                      output_activation):
        if net_type=='conv_net':
            layer_params = [
                ('conv2d', input_dim,32,8,4,0, 'relu'),  
                ('conv2d', 32,64,4,2,0, 'relu'),
                ('conv2d', 64,64,3,1,0, 'relu'),
                ('flatten',None,None),
                ('linear',3136,500,'relu'),
                ('linear',500,300,'relu'),
                ('linear',300,100,'relu'),
                ('linear',100,ouput_dim,output_activation)
                ]
            
        elif net_type=='dense_net':
            layer_params = [
                ('linear',input_dim,400,'relu'),
                ('linear',400,350,'relu'),
                ('linear',350,250,'relu'),
                ('linear',250,100,'relu'),
                ('linear',100,ouput_dim,output_activation)
                ]
        else:
            raise ValueError(f"Unsupported layer type: {net_type}")
        return layer_params
    
    def get_layer(self, layer_desc):
        print(layer_desc)
        if layer_desc[0] == 'linear':
            return nn.Linear(in_features=layer_desc[1], 
                             out_features=layer_desc[2])
        elif layer_desc[0] == 'conv2d':
            return nn.Conv2d(in_channels=layer_desc[1], 
                             out_channels=layer_desc[2], 
                             kernel_size=layer_desc[3], 
                             stride=layer_desc[4], 
                             padding=layer_desc[5])
        elif layer_desc[0]=="flatten":
            return nn.Flatten()
        else:
            raise ValueError(f"Unsupported layer type: {layer_desc[0]}")

    def get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'softmax':
            return nn.Softmax()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def feat_foward(self,x):
        if len(self.layers)%2==0:
            for layer in self.layers[0:-2]:
                x = layer(x)
            return x
        else:
            for layer in self.layers[0:-1]:
                x = layer(x)
            return x

# # Example usage with convolutional layers:
# layer_params = [
#     ('linear',28,400,'relu'),
#     ('linear',400,350,'relu'),
#     ('linear',350,250,'relu'),
#     ('linear',250,100,'relu'),
#     ('linear',100,56,"softmax")
#     ]

# input_data = torch.randn(64, 28) # Example input data (batch size: 64, channels: 3, height: 28, width: 28)
# net = CustomNetwork(layer_params)
# output = net(input_data)
# print(output.shape)  # This will print the shape of the output tensor after passing through all layers and activations.

# net =CustomNetwork(None,True,28,56,"dense_net",'softmax')
# CustomNetwork(None,True,28,56,"conv_net",'softmax')
# CustomNetwork(layer_params,False,None,None,None,None)



# input_data = torch.randn(64,3,84,84) # Example input data (batch size: 64, channels: 3, height: 28, width: 28)
# net =CustomNetwork(None,True,3,56,"conv_net",'softmax')
# output = net(input_data)
# print(output.shape)

class Network(torch.nn.Module):

    def __init__(self, input_dimension, output_dimension, output_activation=torch.nn.Identity()):
        super(Network, self).__init__()
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=64)
        self.layer_2 = torch.nn.Linear(in_features=64, out_features=64)
        self.output_layer = torch.nn.Linear(in_features=64, out_features=output_dimension)
        self.output_activation = output_activation

    def forward(self, inpt):
        layer_1_output = torch.nn.functional.relu(self.layer_1(inpt))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_activation(self.output_layer(layer_2_output))
        return output
    
    def partial_foward(self,inpt):
        layer_1_output = torch.nn.functional.relu(self.layer_1(inpt))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        return layer_2_output