import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
        def __init__(self, state_size, action_size, seed, hidden_in_dim=128, hidden_out_dim=128):
            super(Actor, self).__init__()
            self.seed = torch.manual_seed(seed)
            #self.input_norm = nn.BatchNorm1d(hidden_in_dim)
            self.fc1 = nn.Linear(state_size,hidden_in_dim)
            self.fc2 = nn.Linear(hidden_in_dim,hidden_out_dim)
            self.fc3 = nn.Linear(hidden_out_dim,action_size)
            self.nonlin = f.leaky_relu #leaky_relu
            self.reset_parameters()
        def reset_parameters(self):
            self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
            self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            
        def forward(self, x):
            # return a vector of the actions
            h1 = self.nonlin(self.fc1(x))
            h2 = self.nonlin(self.fc2(h1))
            h3 = torch.tanh(self.fc3(h2))
            return h3
            
#             # h3 is a 2D vector (a force that is applied to the agent)
#             # we bound the norm of the vector to be between 0 and 10
#             return 10.0*(f.tanh(norm))*h3/norm if norm > 0 else 10*h3
        
class Critic(nn.Module):
        def __init__(self, state_size,action_size, seed, hidden_in_dim=128, hidden_out_dim=128):
            super(Critic, self).__init__()
            self.seed = torch.manual_seed(seed)
            self.input_norm = nn.BatchNorm1d(hidden_in_dim)
            self.fc1 = nn.Linear(state_size,hidden_in_dim)
            #2 agents so action size*2 128+2+2=132
            self.fc2 = nn.Linear(hidden_in_dim+ (action_size*2),hidden_out_dim)
            # critic network simply outputs a number
            self.fc3 = nn.Linear(hidden_out_dim,1)
            self.nonlin = f.leaky_relu #leaky_relu
            self.reset_parameters()
        def reset_parameters(self):
            self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
            self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            
        def forward(self, x, action, action_opp):
            # we pass to the critic the state space and actions from all existing players
            h1 = self.nonlin(self.input_norm(self.fc1(x)))
            h2 = self.nonlin(self.fc2(torch.cat((h1, action, action_opp), dim=1)))
            h3 = (self.fc3(h2))
            
            return h3
