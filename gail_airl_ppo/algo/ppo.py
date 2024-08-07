import torch
from torch import nn
from torch.optim import Adam

from .base import Algorithm
from gail_airl_ppo.buffer import RolloutBuffer, StateTransBuffer, RefBufferOfTragectory
from gail_airl_ppo.network import StateIndependentPolicy, StateFunction, ParamGenerateNet


def calculate_gae(values, rewards, dones, next_values, gamma, lambd):
    # Calculate TD errors.
    deltas = rewards + gamma * next_values * (1 - dones) - values
    # Initialize gae.
    gaes = torch.empty_like(rewards, device=torch.device('cuda'))

    # Calculate gae recursively from behind.
    gaes[-1] = deltas[-1]
    for t in reversed(range(rewards.size(0) - 1)):
        gaes[t] = deltas[t] + gamma * lambd * (1 - dones[t]) * gaes[t + 1]


    return gaes + values, (gaes - gaes.mean()) / (gaes.std() + 1e-8)


class PPO(Algorithm):

    def __init__(self, state_shape, action_shape, device, seed, gamma=0.995,trajectory_length = 100,
                 rollout_length=256, mix_buffer=20, lr_actor=1e-6,
                 lr_critic=1e-6, units_actor=(64, 64), units_critic=(64, 64),
                 epoch_ppo=10, clip_eps=0.2, lambd=0.97, coef_ent=0.0,
                 max_grad_norm=10.0):
        super().__init__(state_shape, action_shape, device, seed, gamma)

        

        self.tragectory_state_length = (2*state_shape[0]+action_shape[0])*trajectory_length


        self.learning_steps_ppo = 0
        self.rollout_length = rollout_length
        self.epoch_ppo = epoch_ppo
        self.clip_eps = clip_eps
        self.lambd = lambd
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm
        self.trajectory_length = trajectory_length
        
    def is_update(self, step):
        # print("now state:", step, "need",self.rollout_length )
        return step % self.rollout_length == 0

    def step(self,  envs,  expert_policy ,step_length, params, action_rate = 1):
        # 使用一组参数，跑出一组轨迹
        # trajectory是一堆状态转移，step里面使用状态转移求出参数变化量
        # 使用变化后的参数去跑expert策略
        # 返回值是调整过的环境中，跑出来的轨迹
        env_number = envs.number_env
        envs.reset_isaacgym_env(range(env_number))
        
        
        # print(trajectory_states.shape)

        if params is not None:
            envs.set_params(params=params)


        next_tragectory = [StateTransBuffer(step_length, self.state_shape, \
                                       self.action_shape, self.device) for _ in range(env_number)]

        
        state = envs.reset()['obs']

        for step in range(step_length):
            if expert_policy is not None:
                action, log_pi = expert_policy.explore(state)
            else:
                action = 0.0*torch.ones((env_number,envs.action_space.shape[0]))
                
            action *= action_rate
            next_state, reward, done, _ = envs.step(action)
            next_state = next_state['obs']
            # print(next_state[0:10,1])
            

            for i in range(env_number):
                next_tragectory[i].append(state[i], action[i], next_state[i])
            state = next_state
        
        return next_tragectory
   
    


    def save_models(self, save_dir):
        pass

