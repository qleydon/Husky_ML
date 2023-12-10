# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 11:57:33 2023

@author: Breach
"""

import torch
import numpy as np 
import torch.nn as nn 
from torch.optim.adam import Adam
from src.HUSKY_RL.Husky_Networks import CustomNetwork
from src.HUSKY_RL.Husky_Networks import DenseNet
import itertools


class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor):
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl(p.log(), m) + self.kl(q.log(), m))

class MAPE(nn.Module):
    def __init__(self):
        super(MAPE, self).__init__()
        self.mae = nn.L1Loss()

    def forward(self, target: torch.tensor, prediction: torch.tensor):
        target, prediction = target.view(-1, target.size(-1)), prediction.view(-1, prediction.size(-1))
        return (self.mae(target, prediction))/(torch.abs(target)+torch.finfo(torch.float32).eps)



class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self,batch_size, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon
        self.moving_avg_list=[]
        self.batch_size=batch_size
        
    def update(self, x):
        batch_mean, batch_std, batch_count = np.mean(x), np.std(x), self.batch_size#x.shape[0]
        batch_var = np.square(batch_std)
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.std=self.var**(1/2)
        self.count = new_count
        
    def moving_norm(self,x):
        x=(x-self.mean)/self.std
        # x=x*.9999
        return x


class RNDTauSetter():
    
    
    def __init__(self,
                 input_dim,
                 loss_type,
                 initial_tau,
                 output_dim=2000,
                 learning_rate=.0003,
                 proportion_of_exp_used_for_predictor_update=1,
                 caution=True
                 ):
        
        self.decay_rate= 0.5
        self.increase_factor = 1
        self.current_trace=0
        self.max_error=1

        self.lr=learning_rate
        self.current_lr=learning_rate
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.loss = self._get_loss_function(loss_type)
        self.initial_tau=initial_tau
        self.current_tau=initial_tau
        self._set_networks(input_dim)
        self.caution=caution
        self.proportion_of_exp_used_for_predictor_update=proportion_of_exp_used_for_predictor_update
    
        
    def _get_loss_function(self,loss_type):
        if loss_type == 'KL':
            self.activation_type="softmax"
            return torch.nn.KLDivLoss(reduction="none")
        elif loss_type == 'MSE':
            self.activation_type=None
            return nn.MSELoss(reduction="none")
        elif loss_type == 'MAE':
            self.activation_type=None
            return nn.L1Loss
        elif loss_type == 'MAPE':
            self.activation_type=None
            return MAPE()
        elif loss_type == 'JSD':
            self.activation_type="softmax"
            return JSD()
        else:
            raise ValueError(f"Unsupported loss function: {loss_type}")
        
        
    def  _set_networks(self,inputdim,output_dim=2000):
        self.target=DenseNet(inputdim,output_dim,self.activation_type)
        self.predictor_g=DenseNet(inputdim,output_dim,self.activation_type)
        self.predictor_l=DenseNet(inputdim,output_dim,self.activation_type)
        self.predictor_g.load_state_dict(self.predictor_l.state_dict())
        
        self.predictor_g_opt=Adam(self.predictor_g.parameters(), lr=self.lr)
        self.predictor_l_opt=Adam(self.predictor_l.parameters(), lr=self.lr)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.target.to(self.device)
        self.predictor_g.to(self.device)
        self.predictor_l.to(self.device)
        
    def calculate_errors(self,states_global,states_local):
        with torch.no_grad():
            y_global=self.target(states_global)  
            y_local=self.target(states_local) 
            
            prediction_global=self.predictor_g(states_global)
            prediction_local=self.predictor_l(states_local)
            
            
            global_error =self.loss(y_global,prediction_global)
            global_error=torch.mean(global_error)
            
            local_error =self.loss(y_local,prediction_local)
            local_error=torch.mean(local_error)
        return global_error,local_error
        
        
        
    def apply_mask(self,loss):
        size=loss.size()
        shape=np.asarray(size)
        global_mask=torch.FloatTensor(*shape).uniform_(0, 1).to(self.device)
        global_mask=(global_mask < self.proportion_of_exp_used_for_predictor_update).type(torch.float32)
        loss=torch.sum(loss*global_mask)/torch.max(torch.sum(global_mask),torch.ones(1,device=self.device))
        return loss



    def update_rnd(self,encoding_global, encoding_local):
        with torch.no_grad():
            noisy_targets_global=self.target(encoding_global)  
            noisy_targets_local=self.target(encoding_local) 
        
        global_prediction=self.predictor_g(encoding_global)
        local_prediction=self.predictor_l(encoding_local)
        

        global_loss=self.loss(noisy_targets_global,global_prediction)#**(1/2)
        local_loss=self.loss(noisy_targets_local,local_prediction)

        global_loss=self.apply_mask(global_loss)
        local_loss=self.apply_mask(local_loss)
        # print(local_loss.detach().numpy())
        
        local_loss=torch.mean(local_loss)
        global_loss=torch.mean(global_loss)
        
        # print(local_loss.detach().numpy())
        self.predictor_g_opt.zero_grad()
        global_loss.backward()
        self.predictor_g_opt.step()
        
        self.predictor_l_opt.zero_grad()
        local_loss.backward()
        self.predictor_l_opt.step()
        
        error=global_loss-local_loss
        
        if self.caution:
            self.caution_lr(error)

        return global_loss,local_loss,error
    def step(self, error):
      """
      Pass in the current trace value and the error to compute the next trace value
      :return: The new trace value
      """
      
      error=torch.abs(error)
      error=error.detach().cpu().numpy()
      if self.max_error<error:
          self.max_error=error
      error=error/self.max_error
      # print("this is the error "+str(error))
      delta = -self.decay_rate * self.current_trace + self.increase_factor * error
      self.current_trace=self.current_trace + delta
      # print("this is the current trace "+ str(self.current_trace))
    

    def caution_lr(self,error):
        self.step(error)
        if self.current_trace > .5:
            self.current_lr = (1-self.current_trace)*self.current_lr
            self.predictor_g_opt.param_groups[0]['lr']=self.current_lr
            self.predictor_l_opt.param_groups[0]['lr']=self.current_lr
        else:
            self.current_lr = self.lr
            self.predictor_g_opt.param_groups[0]['lr']=self.current_lr
            self.predictor_l_opt.param_groups[0]['lr']=self.current_lr


    def norm_tau_setter_tf(self,Error_0,alpha,beta,current_error):
        """
        Parameters
        ----------
        T_0 : Initial Tau
            Stable known Env Tau.
        Error_0 : Initial Error
            Firts calculated error of the RND agent .
        alpha : Hyper Parameter
            DESCRIPTION.
        beta : Hyper Parameter
            DESCRIPTION.
        current_error : New Error
            Error of the current predictor of RND.

        Returns
        -------
        Tau.

        """
        num=2*self.initial_tau
        denom=1+torch.pow(Error_0/(alpha*current_error), torch.log(Error_0))
        Tau=num/denom+beta
        return Tau
        
    def save_model(self,file_path):
        torch.save(self.target.state_dict(), file_path+"\\target")
        torch.save(self.predictor_g.state_dict(), file_path+"\\predictor_g")
        torch.save(self.predictor_l.state_dict(), file_path+"\\predictor_l")



    def load_model(self,file_path):
        self.target.load_state_dict(torch.load(file_path+"\\target"))
        self.predictor_g.load_state_dict(torch.load(file_path+"\\predictor_g"))
        self.predictor_l.load_state_dict(torch.load(file_path+"\\predictor_l"))
        
        self.target.eval()  
        self.predictor_g.eval()
        self.predictor_l.eval()
        
    # def tau_setter_rv(self,base_tau,states_global,states_local,prior_shift,prior_tau):
    #    global_error,local_error= self.calculate_error(states_global, states_local)
    #    n_local,n_global=self.norm_error(local_error),self.norm_error(local_error)
    #    env_shift=n_local-n_global
    #    novelty_shift=(env_shift)*self.base_tau
    #    tau=prior_tau+novelty_shift
    #    return tau
       
       


# learning_rates=[]

# # # Example usage with convolutional layers:
# # layer_params = [
# #     ('linear',28,400,'relu'),
# #     ('linear',400,350,'relu'),
# #     ('linear',350,250,'relu'),
# #     ('linear',250,100,'relu'),
# #     ('linear',100,56,"softmax")
# #     ]

# input_data = torch.randn(64, 28) # Example input data (batch size: 64, channels: 3, height: 28, width: 28)
# # net = CustomNetwork(layer_params)
# # output = net(input_data)
# # print(output.shape)  # This will print the shape of the output tensor after passing through all layers and activations.

# # net =CustomNetwork(None,True,28,56,"dense_net",'softmax')
# # CustomNetwork(None,True,28,56,"conv_net",'softmax')
# # CustomNetwork(layer_params,False,None,None,None,None)
# test_lr=[.0003,.5*.0003,.25*.0003]
# for i in test_lr:
#     tausetter=RNDTauSetter(28,"MAPE",.01,learning_rate=i)
#     all_local=[]
#     all_global=[]
    
#     for i in range(65):
#        global_loss,local_loss= tausetter.update_rnd(input_data,input_data)
#        global_loss=global_loss.detach().numpy()
#        local_loss=local_loss.detach().numpy()
#        all_local.append(local_loss)
#     all_local=all_local/all_local[0]
#     learning_rates.append(all_local)

   

# import matplotlib.pyplot as plt

# def plot_three_lists(list1, list2, list3):
#     """
#     Plot three lists of data using Matplotlib.

#     Parameters:
#         list1 (list): The first list of data points.
#         list2 (list): The second list of data points.
#         list3 (list): The third list of data points.
#     """
#     if len(list1) != len(list2) or len(list1) != len(list3):
#         raise ValueError("All lists should have the same length.")

#     x_values = range(len(list1))  # Assuming the x-axis represents indices of the lists.

#     plt.plot(x_values, list1, label='baseline')
#     plt.plot(x_values, list2, label='50%  of baseline')
#     plt.plot(x_values, list3, label='25%  of baseline')

#     plt.xlabel('Training Steps')
#     plt.ylabel('Loss')
#     plt.title('Plot of effects of learning rate on RND')
#     plt.legend()
#     plt.grid(True)
#     plt.show()


# plot_three_lists(learning_rates[0], learning_rates[1], learning_rates[2])