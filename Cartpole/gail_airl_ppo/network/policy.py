import torch
from torch import nn

from .utils import build_mlp, reparameterize, evaluate_lop_pi
from torch.distributions import Categorical

class StateIndependentPolicy(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(64, 64),
                 hidden_activation=nn.Tanh()):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )
        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0]))

    def forward(self, states):
        return torch.tanh(self.net(states))

    def sample(self, states):
        # return reparameterize(self.net(states), self.log_stds)
        return reparameterize(self.forward(states), self.log_stds)

    def evaluate_log_pi(self, states, actions):
        # return evaluate_lop_pi(self.net(states), self.log_stds, actions)
        return evaluate_lop_pi(self.forward(states), self.log_stds, actions)


class StateDependentPolicy(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(256, 256),
                 hidden_activation=nn.ReLU(inplace=True)):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=2 * action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def forward(self, states):
        return torch.tanh(self.net(states).chunk(2, dim=-1)[0])

    def sample(self, states):

        means, log_stds = self.net(states).chunk(2, dim=-1)
        
        temp = reparameterize(means, log_stds.clamp(-20, 2))

        return reparameterize(means, log_stds.clamp(-20, 2))




class ParamGenerateNet(nn.Module):

    def __init__(self, state_shape, action_shape):
        super().__init__()

        input_dim=state_shape[0]
        output_dim=action_shape[0]
        self.fc1=nn.Linear(input_dim, 256)
        self.fc2=nn.Linear(256, 64)
        # self.mix=nn.Linear(output_dim+64, output_dim*3)
        self.mix=nn.Linear(64, output_dim*3)
        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0]))

    def forward(self, states, now_param):
        proc1 = torch.relu(self.fc1(states))
        proc2 = torch.relu(self.fc2(proc1))
        
        # mixed  = self.mix(torch.cat([proc2, now_param],dim=-1))
        mixed  = self.mix(proc2)

        return torch.softmax(mixed,dim=-1)


    # action, log_pi = self.actor.sample(state, now_param)
    def sample(self, states, now_param):
        # return reparameterize(self.net(states), self.log_stds)
        act_prob = self.forward(states, now_param)
         
        m = Categorical(act_prob)
        actions = m.sample().unsqueeze(dim=-1)
        # print(act_prob)
        # print(torch.gather(act_prob, -1, actions) )

        return actions,torch.gather(act_prob, -1, actions) 

    def calculate_pi(self, states, now_param, actions):
        # return evaluate_lop_pi(self.net(states), self.log_stds, actions)
        act_prob = self.forward(states, now_param)
    
        return torch.gather(act_prob, -1, actions)
    

# if __name__ == "__main__":
#     p1 = ParamGenerateNet((5000,1), (1,1))
#     x1 = torch.rand((10, 5000))
#     x2 = torch.rand((10,1))
#     print(p1.forward(x1, x2))


    # p2 = StateIndependentPolicy((5000,), (1,))
    # x1_ = torch.rand((10, 5000))
    # print(p2.sample(x1_))