import os
import argparse
from datetime import datetime
import numpy as np

# from gail_airl_ppo.env import make_env
from isaac_gym_env import paramAnt

from gail_airl_ppo.algo import  SACExpert
from torch.utils.tensorboard import SummaryWriter

import torch

import re
import time
from gail_airl_ppo.algo import GARAT




DEVICE = 'cuda:0'
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main(args):
    setup_seed(args.seed)
    PRESET_PARMAS =  [1.5, 0.3,   0.2, 0.3, 0.1,   0.1, 0.2, 0.1,  0.1, 0.2, 1]            #Ant参数
   

    envs = paramAnt(args.number_of_env, headless=False , seed=args.seed,rl_device=DEVICE)


    envs.set_params(params=PRESET_PARMAS)
    
    

    state_shape = (*envs.observation_space.shape, 1)
    action_shape = (*envs.action_space.shape, 1)
    buffer_real = None

    path = args.actor_weight
  

    relative_path = path

    if not os.path.isdir(relative_path): #判断是否是文件夹，不是文件夹才打开
        actor_policy = SACExpert(
            state_shape=state_shape,
            action_shape=action_shape,
            device=torch.device(DEVICE),
            path=relative_path)
        reward = evaluate(envs, actor_policy, args.evaluate_steps, args.number_of_env)


def evaluate(envs, actor_policy, evaluate_steps, number_of_env):

    GAT_ANT = GARAT(buffer_exp=None, state_shape=envs.observation_space.shape, 
                    action_shape=envs.action_space.shape, number_of_envs=1, 
                    device=DEVICE,seed=args.seed, action_policy_dir = None )
    GAT_ANT.load_actor(args.GAT_policy)


    now_step = 0
    env_steps = [0 for i in range(number_of_env)]
    state = envs.reset()['obs']

    sum_reward = None
    episode_times = 0
    final_reward = 0
    while now_step < evaluate_steps:

        # Pass to the algorithm to update state and episode timestep.

        now_step += number_of_env
        # t0 = time.time_ns()
        state, rewards, done = actor_policy.step(envs, number_of_env, state, GAT_ANT)
        # t1 = time.time_ns()
        # print((t1-t0)*1e-6)

        if sum_reward is None:
            sum_reward = rewards
        else:
            sum_reward += rewards

        for i in range(number_of_env):
            env_steps[i] += 1
            if done[i] or env_steps[i] >= envs.max_episode_length:
                env_steps[i] = 0
                final_reward += sum_reward[i]

                sum_reward[i] *= 0
                episode_times += 1

 


    # print(rewards.mean().item())

    return (final_reward/episode_times).item()



if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--env_id', type=str, default='Ant_GAT')
    p.add_argument('--cuda', default=True ,action='store_true')

    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--actor_weight', type=str, default='logs/Ant_GAT_pure/SAC_baseline/0.5/seed0-20240805-1008/model/isaac_step10000000/actor.pth')
    p.add_argument('--summary_dir', type=str, default='logs/')
    p.add_argument('--tag', type=str, default='SAC_DR')
    p.add_argument('--number_of_env', type=int, default=100)
    p.add_argument('--evaluate_steps', type=int, default=50000)
    p.add_argument('--GAT_policy', type=str, default='logs/Ant_GAT_pure/GAT_OOD/OOD05/seed1/model/isaac_step42000000/actor.pth')

    args = p.parse_args()


    main(args)


