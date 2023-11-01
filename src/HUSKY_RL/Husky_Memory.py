# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 13:25:19 2023

@author: Breach
"""

import numpy as np
import os 

class ReplayBuffer:

    def __init__(self, environment, capacity=5000):
        self.atari_setting=False
        transition_type_str = self.get_transition_type_str(environment)
        self.buffer = np.zeros(capacity, dtype=transition_type_str)
        self.weights = np.zeros(capacity)
        self.head_idx = 0
        self.count = 0
        self.capacity = capacity
        self.max_weight = 10**-2
        self.delta = 10**-4
        self.indices = None
        

    def get_transition_type_str(self, environment):
        
        
        state_dim = 26 #environment.observation_size
        state_dim_str = '' if state_dim == () else str(state_dim)
        state_type_str = 'float32' #str(environment.observation_space.dtype)
        action_dim = 5# environment.action_size
        action_dim_str = '' if action_dim == () else str(action_dim)
        action_type_str = 'int64' # str(environment.action_space.dtype)    

        # type str for transition = 'state type, action type, reward type, state type'
        transition_type_str = '{0}{1}, {2}{3}, float32, {0}{1}, bool'.format(state_dim_str, state_type_str,
                                                                             action_dim_str, action_type_str)
        
        print("transition_type_str: ", transition_type_str)
        #transition_type_str = '26float32, float32, int64, 5int64, bool'
        #print("transition_type_str: ", transition_type_str)
        
        return transition_type_str

    def add_transition(self, transition):
        self.buffer[self.head_idx] = transition
        self.weights[self.head_idx] = self.max_weight

        self.head_idx = (self.head_idx + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample_minibatch(self, size=100):
        set_weights = self.weights[:self.count] + self.delta
        probabilities = set_weights / sum(set_weights)
        self.indices = np.random.choice(range(self.count), size, p=probabilities, replace=False)
        return self.buffer[self.indices]

    def update_weights(self, prediction_errors):
        max_error = max(prediction_errors)
        self.max_weight = max(self.max_weight, max_error)
        self.weights[self.indices] = prediction_errors
        
    def sample_latest(self,size=100):
        indices=list(range(self.count-100,self.count))
        return self.buffer[indices]

    def get_size(self):
        return self.count

    def save_state(self,path="AgentReplay"):
        if not os.path.exists(path):
            os.mkdir(path)
        np.save(path+"\\agents_data",self.buffer)
        
    def load_state(self,path):
        self.buffer=np.load(path+".npy")
        