import torch
from torch import nn
import torch.nn.functional as F

from .utils import build_mlp


class GAILDiscrim(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(100, 100),
                 hidden_activation=nn.ReLU()):
        super().__init__()
        
        self.net = build_mlp(
            input_dim=2*state_shape[0] + action_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def forward(self, states, actions, next_states):
        # print(states.shape, actions.shape, next_states.shape)
        # temp = torch.cat([states, actions, next_states], dim=1).squeeze()
        # temp = self.net(torch.cat([states, actions, next_states], dim=1).squeeze())
        return self.net(torch.cat([states, actions, next_states], dim=1).squeeze())



    def calculate_reward_WGail(self, tragectory):
        # PPO(GAIL) is to maximize E_{\pi} [-log(1 - D)].

        rewards = torch.zeros((len(tragectory),1), device=torch.device('cuda'))
        # print(rewards.shape)
        for i in range(len(tragectory)):
            states,actions,next_states = tragectory[i].get()
      

            with torch.no_grad():
                single_reward =  self.forward(states, actions, next_states)
                # print(single_reward.shape)
            rewards[i] = single_reward.squeeze().mean()
        return rewards

        


class AIRLDiscrim(nn.Module):

    def __init__(self, state_shape, gamma,
                 hidden_units_r=(64, 64),
                 hidden_units_v=(64, 64),
                 hidden_activation_r=nn.ReLU(inplace=True),
                 hidden_activation_v=nn.ReLU(inplace=True)):
        super().__init__()

        self.g = build_mlp(
            input_dim=state_shape[0],
            output_dim=1,
            hidden_units=hidden_units_r,
            hidden_activation=hidden_activation_r
        )
        self.h = build_mlp(
            input_dim=state_shape[0],
            output_dim=1,
            hidden_units=hidden_units_v,
            hidden_activation=hidden_activation_v
        )

        self.gamma = gamma

    def f(self, states, dones, next_states):
        rs = self.g(states)
        vs = self.h(states)
        next_vs = self.h(next_states)
        return rs + self.gamma * (1 - dones) * next_vs - vs

    def forward(self, states, dones, log_pis, next_states):
        # Discriminator's output is sigmoid(f - log_pi).
        return self.f(states, dones, next_states) - log_pis

    def calculate_reward(self, states, dones, log_pis, next_states):
        with torch.no_grad():
            logits = self.forward(states, dones, log_pis, next_states)
            return -F.logsigmoid(-logits)
