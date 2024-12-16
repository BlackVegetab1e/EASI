import os
import sys

# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取当前文件的目录
current_dir = os.path.dirname(current_file_path)
# 获取上一级目录
parent_dir = os.path.dirname(current_dir)
# 将上一级目录添加到sys.path中
sys.path.append(parent_dir)
import argparse
from datetime import datetime
import numpy as np

# from gail_airl_ppo.env import make_env
from isaac_gym_env import paramAnt, paramCartpoleFull, paramBallBalance

from EvolutionaryAdversarial.algo import StateTransDiscriminator, SACExpert

from EvolutionaryAdversarial.buffer import SerializedBuffer, RefBufferOfTragectory
import torch

DEVICE = 'cuda:0'




def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)



def collect_demo(args):
    setup_seed(args.seed)

    if args.env_id == 'Ant': 
        envs = paramAnt(args.number_of_env, DEVICE, seed=args.seed, headless=True)
        PRESET_PARMAS = [1.5, 0.3,   0.2, 0.3, 0.1,   0.1, 0.2, 0.1,  0.1, 0.2, 1] 
        if(args.OOD):    
            # Body mass
            PRESET_PARMAS[10]*=args.OOD_rate

    elif args.env_id == 'Ballbalance':
        envs = paramBallBalance(args.number_of_env, DEVICE, seed=args.seed, headless=True) 
        PRESET_PARMAS = [3,5,1,0.3,100,10,5]
        if(args.OOD):
            # Ball mass
            PRESET_PARMAS[0]*=args.OOD_rate
    
    elif args.env_id == 'Cartpole':
        envs = paramCartpoleFull(args.number_of_env, DEVICE, seed=args.seed, headless=True)  
        PRESET_PARMAS = [0.3, 0.1, 0.3, 3e-04, 2e-03 ,5e-03, 1e-02, 20, 0.3, 5, 0.6]
        if(args.OOD):
            # Cart P
            PRESET_PARMAS[7]*=args.OOD_rate
   
    envs.set_params(params=PRESET_PARMAS)

    ood_tag = 'WD'
    if(args.OOD):
        ood_tag = f'OOD{args.OOD_rate}'
    save_dir = f'{args.summary_dir}/{args.env_id}/demonstration/{ood_tag}/size{args.collect_steps}_traj_length{args.trajectory_length}_real_domain_cpu_seed_{args.seed}.pth'



    state_shape = (*envs.observation_space.shape, 1)
    action_shape = (*envs.action_space.shape, 1)
    buffer_real = None


    expert_policy = SACExpert(
        state_shape=state_shape,
        action_shape=action_shape,
        device=torch.device(DEVICE),
        path=args.expert_weight
    )


    gail = StateTransDiscriminator(
    buffer_exp=buffer_real,
    state_shape=state_shape,
    action_shape=action_shape,
    device=torch.device(DEVICE),
    )
    isaac_step = 0
    
    ref_tragectory_buffer = RefBufferOfTragectory(args.collect_steps, torch.device(DEVICE))
    
    while isaac_step < args.collect_steps:

        tragectory_pop = gail.step(envs, expert_policy=expert_policy, step_length = args.trajectory_length, params=None)

        for i in range(len(tragectory_pop)):
            ref_tragectory_buffer.append(tragectory_pop[i])
            # print(tragectory_pop[i])
            isaac_step += args.trajectory_length
            
        
        print(isaac_step)

    ref_tragectory_buffer.save(save_dir, args.trajectory_length)



if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--env_id', type=str, default='Ant')
    p.add_argument('--cuda', default=True ,action='store_true')
    p.add_argument('--seed', type=int, default=2)
    p.add_argument('--summary_dir', type=str, default='logs/')
    p.add_argument('--expert_weight', type=str, default='logs/Ant/SAC_DR_test/seed2-20240419-1654/final_model/actor.pth')
    p.add_argument('--trajectory_length', type=int, default=200)
    p.add_argument('--OOD', action='store_true', default=False)
    p.add_argument('--OOD_rate', type=float, default=1)
    p.add_argument('--number_of_env', type=int, default=100)
    p.add_argument('--collect_steps', type=int, default=40000)
    args = p.parse_args()
    collect_demo(args)


