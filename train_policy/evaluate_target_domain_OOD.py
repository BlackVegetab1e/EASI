import os
import argparse
from datetime import datetime
import numpy as np

# from gail_airl_ppo.env import make_env
from isaac_gym_env import paramAnt

from EvolutionaryAdversarial.algo import  SACExpert
from torch.utils.tensorboard import SummaryWriter

import torch

import re
import time





DEVICE = 'cuda:0'
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main(args):
    setup_seed(args.seed)
    PRESET_PARMAS =  [1.5, 0.3,   0.2, 0.3, 0.1,   0.1, 0.2, 0.1,  0.1, 0.2, 1]            #Ant参数

    if(args.OOD):
        PRESET_PARMAS[10] *= args.OOD_rate

    envs = paramAnt(args.number_of_env, headless=True , seed=args.seed,rl_device=DEVICE)

    summary_log_dir = args.summary_dir + args.env_id +'/real_domain_test/' +args.tag+'/seed_'+str(args.seed)+'OOD_'+str(args.OOD_rate)
    writer = SummaryWriter(log_dir= summary_log_dir)

    envs.set_params(params=PRESET_PARMAS)
    
    

    state_shape = (*envs.observation_space.shape, 1)
    action_shape = (*envs.action_space.shape, 1)
    buffer_real = None

    path = args.actor_weight_dir
    files= os.listdir(path)
    iter_number = [int(re.findall(r'\d+', file)[-1]) for file in files]

    sort_index = np.argsort(iter_number)

    # for i in sort_index:
    #     print(i)
    #     print(files[i] )
    # print(files)

    files = [files[i] for i in sort_index]
    print('There are',len(files),'actors to evaluate')
    file = files[-1]
    
    iter = re.findall(r'\d+', file)
    iter = [int(n) for n in iter]  # 转换为整数

    relative_path = path+"/"+file+'/actor.pth'

    if not os.path.isdir(relative_path): #判断是否是文件夹，不是文件夹才打开
        actor_policy = SACExpert(
            state_shape=state_shape,
            action_shape=action_shape,
            device=torch.device(DEVICE),
            path=relative_path)
        reward = evaluate(envs, actor_policy, args.evaluate_steps, args.number_of_env)
    print(iter[-1],':', reward)
    writer.add_scalar('reward', reward, iter[-1])




def evaluate(envs, actor_policy, evaluate_steps, number_of_env):

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
        state, rewards, done = actor_policy.step(envs, number_of_env, state)
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
    """
    针对OOD设计的实验,只需要测试最后一个即可
    """


    p = argparse.ArgumentParser()
    p.add_argument('--env_id', type=str, default='Ant')
    p.add_argument('--cuda', default=True ,action='store_true')
    # p.add_argument('--seed', type=int, default=0)
    # p.add_argument('--actor_weight_dir', type=str, default='logs/Ant/SAC_DR_search/seed0-20240419-1907/model')
    # p.add_argument('--seed', type=int, default=1)
    # p.add_argument('--actor_weight_dir', type=str, default='logs/Ant/SAC_DR_search/seed1-20240419-1945/model')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--actor_weight_dir', type=str, default='logs/Ant_OOD/SAC_DR_search/OOD_RATE_0.5/seed0-20240512-2159/model')
    p.add_argument('--summary_dir', type=str, default='logs/')
    p.add_argument('--tag', type=str, default='SAC_DR_search')
    p.add_argument('--trajectory_length', type=int, default=200)
    p.add_argument('--number_of_env', type=int, default=100)
    p.add_argument('--evaluate_steps', type=int, default=20000)
    
    p.add_argument('--OOD', type=bool, default=True)
    p.add_argument('--OOD_rate', type=float, default=1)
    args = p.parse_args()

    if(args.OOD):
        args.env_id = args.env_id + '_OOD'

    main(args)


