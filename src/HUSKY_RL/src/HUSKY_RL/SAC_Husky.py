#!/usr/bin/env python

import rospy
import os
import time
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from collections import deque
from std_msgs.msg import Float32MultiArray
from src.HUSKY_RL.Husky_ENV import Env
import torch.nn.functional as F
import torch.optim as optim


from src.HUSKY_RL.Husky_RNDUtils import RNDTauSetter
import torch
import numpy as np
from src.HUSKY_RL.Husky_Networks import CustomNetwork
from src.HUSKY_RL.Husky_Memory import ReplayBuffer
import os


class SACAgent:

    ALPHA_INITIAL = 1.
    REPLAY_BUFFER_BATCH_SIZE = 256
    DISCOUNT_RATE = 0.99
    LEARNING_RATE = 10 ** -4
    SOFT_UPDATE_INTERPOLATION_FACTOR = 0.01

    def __init__(self,environment,rnd):
        self.rnd_state_flag=False
        self.rnd_state_only=False
        self.environment = environment
        self.state_dim = 3 # 3 channels?
        self.action_dim = self.environment.action_size
        if(self.action_dim == None):
            self.action_dim = 7 #5
        self.critic_local = CustomNetwork(None,True,self.state_dim,
                                          self.action_dim,"conv_net",None)

        self.critic_local2 = CustomNetwork(None,True,self.state_dim,
                                          self.action_dim,"conv_net",None)

        self.critic_optimiser = torch.optim.Adam(self.critic_local.parameters(), lr=self.LEARNING_RATE)
        self.critic_optimiser2 = torch.optim.Adam(self.critic_local2.parameters(), lr=self.LEARNING_RATE)

        self.critic_target = CustomNetwork(None,True,self.state_dim,
                                          self.action_dim,"conv_net",None)

        self.critic_target2 =CustomNetwork(None,True,self.state_dim,
                                          self.action_dim,"conv_net",None)


        self.soft_update_target_networks(tau=1.)

        self.actor_local = CustomNetwork(None,True,self.state_dim,
                                          self.action_dim,"conv_net",'softmax')
        
        self.actor_optimiser = torch.optim.Adam(self.actor_local.parameters(), lr=self.LEARNING_RATE)

        self.replay_buffer = ReplayBuffer(self.environment,atari=True)
        self.rnd_flag=rnd
        if rnd:
            self._set_rnd()
        self.target_entropy = 0.98 * -np.log(1 / self.environment.action_size)
        self.log_alpha = torch.tensor(np.log(self.ALPHA_INITIAL), requires_grad=True)
        self.alpha = self.log_alpha
        self.alpha_optimiser = torch.optim.Adam([self.log_alpha], lr=self.LEARNING_RATE)
    
    def _set_rnd(self):
        output=self.critic_local.feat_foward(torch.rand((64,self.state_dim,86,86)))
        input_dim=2*(output.shape[1])
        loss_type="MAPE"
        input_dim2=2*(self.state_dim)
        self.tausetter=RNDTauSetter(input_dim,loss_type,initial_tau=.003)
        self.tausetter2=RNDTauSetter(input_dim2,loss_type,initial_tau=.003)
    def get_next_action(self, state, evaluation_episode=False):
        if evaluation_episode:
            discrete_action = self.get_action_deterministically(state)
        else:
            discrete_action = self.get_action_nondeterministically(state)
        return discrete_action

    def get_action_nondeterministically(self, state):
        action_probabilities = self.get_action_probabilities(state)
        discrete_action = np.random.choice(range(self.action_dim), p=action_probabilities)
        return discrete_action

    def get_action_deterministically(self, state):
        action_probabilities = self.get_action_probabilities(state)
        discrete_action = np.argmax(action_probabilities)
        return discrete_action
    
    def train_on_transition(self, state, discrete_action, next_state, reward, done):
        #print("state_shape", state.shape())
        #print("discrete_action", discrete_action)
        #print("next_state.shape", next_state.shape())
        #print("reward", reward)
        #print("done", done)
        transition = (state, discrete_action, reward, next_state, done)
        g_loss,l_loss,error=self.train_networks(transition)
        return g_loss,l_loss,error
    
    def train_networks(self, transition):
        # Set all the gradients stored in the optimisers to zero.
        self.critic_optimiser.zero_grad()
        self.critic_optimiser2.zero_grad()
        self.actor_optimiser.zero_grad()
        self.alpha_optimiser.zero_grad()
        # Calculate the loss for this transition.
        
        self.replay_buffer.add_transition(transition)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network
        # parameters.
        if self.replay_buffer.get_size() >= self.REPLAY_BUFFER_BATCH_SIZE:
            # get minibatch of 100 transitions from replay buffer
            minibatch = self.replay_buffer.sample_minibatch(self.REPLAY_BUFFER_BATCH_SIZE)
            local_minibatch=self.replay_buffer.sample_minibatch(self.REPLAY_BUFFER_BATCH_SIZE)
            minibatch_separated = list(map(list, zip(*minibatch)))
            local_minibatch_separated = list(map(list, zip(*local_minibatch)))

            # unravel transitions to get states, actions, rewards and next states
            states_tensor = torch.tensor(np.array(minibatch_separated[0]))
            actions_tensor = torch.tensor(np.array(minibatch_separated[1]))
            rewards_tensor = torch.tensor(np.array(minibatch_separated[2])).float()
            next_states_tensor = torch.tensor(np.array(minibatch_separated[3]))
            done_tensor = torch.tensor(np.array(minibatch_separated[4]))
            
            states_tensor_local = torch.tensor(np.array(local_minibatch_separated[0]))
            next_states_tensor_local = torch.tensor(np.array(local_minibatch_separated[3]))
            
            
            critic_loss, critic2_loss = \
                self.critic_loss(states_tensor, actions_tensor, rewards_tensor, next_states_tensor, done_tensor)
            
            if self.rnd_flag:
                
                if self.rnd_state_flag:
                    global_loss,local_loss,error,global_state_loss,local_state_loss,error_state=self.rnd_loss_states(states_tensor,next_states_tensor,
                                                          states_tensor_local,next_states_tensor_local)
                elif self.rnd_state_only:
                    global_loss,local_loss,error=self.rnd_state_loss(states_tensor,next_states_tensor,
                                                          states_tensor_local,next_states_tensor_local)
                
                else:
                    global_loss,local_loss,error=self.rnd_loss(states_tensor,next_states_tensor,
                                                          states_tensor_local,next_states_tensor_local)
                

            
            critic_loss.backward()
            critic2_loss.backward()
            self.critic_optimiser.step()
            self.critic_optimiser2.step()

            actor_loss, log_action_probabilities = self.actor_loss(states_tensor)

            actor_loss.backward()
            self.actor_optimiser.step()

            alpha_loss = self.temperature_loss(log_action_probabilities)

            alpha_loss.backward()
            self.alpha_optimiser.step()
            self.alpha = self.log_alpha.exp()

            self.soft_update_target_networks()
            
            if self.rnd_state_flag:
                return global_loss,local_loss,error,global_state_loss,local_state_loss,error_state
            return global_loss,local_loss,error
        return 0,0,0

    def critic_loss(self, states_tensor, actions_tensor, rewards_tensor, next_states_tensor, done_tensor):
        with torch.no_grad():
            action_probabilities, log_action_probabilities = self.get_action_info(next_states_tensor)
            next_q_values_target = self.critic_target.forward(next_states_tensor)
            next_q_values_target2 = self.critic_target2.forward(next_states_tensor)
            soft_state_values = (action_probabilities * (
                    torch.min(next_q_values_target, next_q_values_target2) - self.alpha * log_action_probabilities
            )).sum(dim=1)

            next_q_values = rewards_tensor + ~done_tensor * self.DISCOUNT_RATE*soft_state_values

        #print("action_tensor ", actions_tensor)
        #print("actions_tensor.dtype ", actions_tensor.dtype)

        soft_q_values = self.critic_local(states_tensor).gather(1, actions_tensor.unsqueeze(-1)).squeeze(-1)
        soft_q_values2 = self.critic_local2(states_tensor).gather(1, actions_tensor.unsqueeze(-1)).squeeze(-1)
        critic_square_error = torch.nn.MSELoss(reduction="none")(soft_q_values, next_q_values)
        critic2_square_error = torch.nn.MSELoss(reduction="none")(soft_q_values2, next_q_values)
        weight_update = [min(l1.item(), l2.item()) for l1, l2 in zip(critic_square_error, critic2_square_error)]
        self.replay_buffer.update_weights(weight_update)
        critic_loss = critic_square_error.mean()
        critic2_loss = critic2_square_error.mean()
        return critic_loss, critic2_loss
    
    def rnd_state_loss(self, states_tensor,states_local_tensor,next_states_tensor,next_states_tensor_local):
        with torch.no_grad():
    
            encoding_global=torch.cat((states_tensor, next_states_tensor), 1)
            encoding_local=torch.cat((states_local_tensor, next_states_tensor_local), 1)
            
        global_loss,local_loss,error=self.tausetter2.update_rnd(encoding_global,encoding_local)
        return global_loss,local_loss,error
    
    def rnd_loss(self, states_tensor,states_local_tensor,next_states_tensor,next_states_tensor_local):
        with torch.no_grad():
            encoding_global= self.critic_local.feat_foward(states_tensor)
            encoding_local = self.critic_local.feat_foward(states_local_tensor)

            encoding_global_next = self.critic_local.feat_foward(next_states_tensor_local)
            encoding_local_next = self.critic_local.feat_foward(next_states_tensor_local)
            
            encoding_global=torch.cat((encoding_global, encoding_global_next), 1)
            encoding_local=torch.cat((encoding_local, encoding_local_next), 1)
            
        global_loss,local_loss,error=self.tausetter.update_rnd(encoding_global,encoding_local)
        
        return global_loss,local_loss,error
    
    def rnd_only_state_loss(self, states_tensor,states_local_tensor,next_states_tensor,next_states_tensor_local):
        with torch.no_grad():
            encoding_global= self.critic_local.feat_foward(states_tensor)
            encoding_local = self.critic_local.feat_foward(states_local_tensor)

            encoding_global_next = self.critic_local.feat_foward(next_states_tensor_local)
            encoding_local_next = self.critic_local.feat_foward(next_states_tensor_local)
            
            encoding_global=torch.cat((encoding_global, encoding_global_next), 1)
            encoding_local=torch.cat((encoding_local, encoding_local_next), 1)
            
        global_loss,local_loss,error=self.tausetter2.update_rnd(encoding_global,encoding_local)
        
        return global_loss,local_loss,error
        

    def rnd_loss_states(self, states_tensor,states_local_tensor,next_states_tensor,next_states_tensor_local):
        with torch.no_grad():
            encoding_global= self.critic_local.feat_foward(states_tensor)
            encoding_local = self.critic_local.feat_foward(states_local_tensor)

            encoding_global_next = self.critic_local.feat_foward(next_states_tensor_local)
            encoding_local_next = self.critic_local.feat_foward(next_states_tensor_local)
            
            encoding_global=torch.cat((encoding_global, encoding_global_next), 1)
            encoding_local=torch.cat((encoding_local, encoding_local_next), 1)
            
            encoding_global_states=torch.cat((states_tensor, next_states_tensor), 1)
            encoding_local_states=torch.cat((states_local_tensor, next_states_tensor_local), 1)
            
        
        global_state_loss,local_state_loss,error_state=self.tausetter2.update_rnd(encoding_global_states, encoding_local_states)
        global_loss,local_loss,error=self.tausetter.update_rnd(encoding_global,encoding_local)
        
        return global_loss,local_loss,error,global_state_loss,local_state_loss,error_state
    
    def actor_loss(self, states_tensor):
        action_probabilities, log_action_probabilities = self.get_action_info(states_tensor)
        q_values_local = self.critic_local(states_tensor)
        q_values_local2 = self.critic_local2(states_tensor)
        inside_term = self.alpha * log_action_probabilities - torch.min(q_values_local, q_values_local2)
        policy_loss = (action_probabilities * inside_term).sum(dim=1).mean()
        return policy_loss, log_action_probabilities

    def temperature_loss(self, log_action_probabilities):
        alpha_loss = -(self.log_alpha * (log_action_probabilities + self.target_entropy).detach()).mean()
        return alpha_loss

    def get_action_info(self, states_tensor):
        action_probabilities = self.actor_local.forward(states_tensor)
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probabilities + z)
        return action_probabilities, log_action_probabilities

    def get_action_probabilities(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_probabilities = self.actor_local.forward(state_tensor)
        return action_probabilities.squeeze(0).detach().numpy()

    def soft_update_target_networks(self, tau=SOFT_UPDATE_INTERPOLATION_FACTOR):
        self.soft_update(self.critic_target, self.critic_local, tau)
        self.soft_update(self.critic_target2, self.critic_local2, tau)

    def soft_update(self, target_model, origin_model, tau):
        for target_param, local_param in zip(target_model.parameters(), origin_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    def predict_q_values(self, state):
        q_values = self.critic_local(state)
        q_values2 = self.critic_local2(state)
        return torch.min(q_values, q_values2)
    
    def save_models(self,file_path):
        if not os.path.exists(file_path):
            os.makedirs(file_path)
            print(file_path)
            
        torch.save(self.critic_local.state_dict(), file_path+"\\critic_local")
        torch.save(self.critic_local2.state_dict(), file_path+"\\critic_local2")
        torch.save(self.actor_local.state_dict(), file_path+"\\actor")
        torch.save(self.critic_target.state_dict(), file_path+"\\t_critic_local")
        torch.save(self.critic_target2.state_dict(), file_path+"\\t_critic_local2")
        
    def load_models(self,file_path):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        self.critic_local.load_state_dict(torch.load(file_path+timestr+"\\critic_local"))
        self.critic_local2.load_state_dict(torch.load(file_path+timestr+"\\critic_local2"))
        self.actor_local.load_state_dict(torch.load(file_path+timestr+"\\actor"))
        self.critic_target.load_state_dict(torch.load(file_path+timestr+"\\t_critic_local"))
        self.critic_target2.load_state_dict(torch.load(file_path+timestr+"\\t_critic_local2"))
        
        self.critic_local.eval()  
        self.critic_local2.eval()
        self.actor_local.eval()
    
        
        
    def save_state(self,path):
        self.replay_buffer.save_state(path)
        self.save_models(path)
        self.tausetter.save_model(path)
        
    def load_state(self,path):
        self.replay_buffer.load_state(path+"\\agents_data")
        self.load_models(path)        
        self.tausetter.load_model(path)     