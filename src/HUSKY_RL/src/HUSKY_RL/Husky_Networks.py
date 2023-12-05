# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 11:19:48 2023

@author: Breach
"""

import torch
import itertools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


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

class CustomNetwork(nn.Module):
    def __init__(self,
                 cnn_input_dim,
                 cnn_ouput_dim,
                 array_input_dim,
                 array_output_dim,
                 siamese_output_dim,
                 output_activation):
        self.cnn_conv2d_1 = nn.Conv2d(cnn_input_dim, 32,8,4,0)
        self.cnn_conv2d_2 = nn.Conv2d(32,64,4,2,0)
        self.cnn_conv2d_3 = nn.Conv2d(64,64,3,1,0)
        self.flatten = nn.Flatten()
        self.cnn_linear = nn.Linear(3136,cnn_ouput_dim)

        self.arr_linear_1 = nn.Linear(array_input_dim, 1000)
        self.arr_linear_2 = nn.Linear(1000, 256)
        self.arr_linear_3 = nn.Linear(256, array_output_dim)

        self.siam_linear_1 = nn.Linear(cnn_ouput_dim + array_output_dim, 500)
        self.siam_linear_2 = nn.Linear(500, 300)
        self.siam_linear_3 = nn.Linear(300, 100)
        self.siam_linear_4 = nn.Linear(100, siamese_output_dim)

        self.activation = get_activation(output_activation)
        
        
    def forward(self, x_cnn, x_array):
        x_cnn = F.relu(self.cnn_conv2d_1(x_cnn))
        x_cnn = F.relu(self.cnn_conv2d_2(x_cnn))
        x_cnn = F.relu(self.cnn_conv2d_3(x_cnn))
        x_cnn = self.flatten(x_cnn)
        x_cnn = F.relu(self.cnn_linear(x_cnn))

        x_array = F.relu(self.arr_linear_1(x_array))
        x_array = F.relu(self.arr_linear_2(x_array))
        x_array = F.relu(self.arr_linear_3(x_array))

        x_siam = torch.concat(x_cnn, x_array)

        x_siam = F.relu(self.siam_linear_1(x_siam))
        x_siam = F.relu(self.siam_linear_2(x_siam))
        x_siam = F.relu(self.siam_linear_3(x_siam))
        x_siam = self.siam_linear_4(x_siam)

        return self.activation(x_siam)

    def feat_forward(self, x_cnn, x_array):
        x_cnn = F.relu(self.cnn_conv2d_1(x_cnn))
        x_cnn = F.relu(self.cnn_conv2d_2(x_cnn))
        x_cnn = F.relu(self.cnn_conv2d_3(x_cnn))
        x_cnn = self.flatten(x_cnn)
        x_cnn = F.relu(self.cnn_linear(x_cnn))

        x_array = F.relu(self.arr_linear_1(x_array))
        x_array = F.relu(self.arr_linear_2(x_array))
        x_array = F.relu(self.arr_linear_3(x_array))

        x_siam = torch.concat(x_cnn, x_array)

        x_siam = F.relu(self.siam_linear_1(x_siam))
        x_siam = F.relu(self.siam_linear_2(x_siam))
        x_siam = F.relu(self.siam_linear_3(x_siam))

        return x_siam