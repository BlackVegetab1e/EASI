import os
import argparse
from datetime import datetime
import numpy as np

# from gail_airl_ppo.env import make_env
from isaac_gym_env import paramAnt

from EvolutionaryAdversarial.algo import StateTransDiscriminator, SACExpert

from EvolutionaryAdversarial.buffer import SerializedBuffer, RefBufferOfTragectory
import torch

DEVICE = 'cuda:0'


# TODO 整合三个部分的collect环节

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)



def collect_demo(args):

    setup_seed(args.seed)
    PRESET_PARMAS = [1.5, 0.3,   0.2, 0.3, 0.1,   0.1, 0.2, 0.1,  0.1, 0.2, 1]       

    envs = paramAnt(args.number_of_env, rl_device=DEVICE, headless=True, seed=args.seed)
    envs.set_params(params=PRESET_PARMAS)



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
    tragectorch_length= args.trajectory_length, 
    device=torch.device(DEVICE),
    seed=0,
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

    ref_tragectory_buffer.save(os.path.join(
        'logs',args.env_id,
        'expert',
        f'size{args.collect_steps}_traj_length{args.trajectory_length}_real_domain_cpu_seed_{args.seed}.pth'
    ))



if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--env_id', type=str, default='Ant')
    p.add_argument('--cuda', default=True ,action='store_true')
    p.add_argument('--seed', type=int, default=2)
    p.add_argument('--expert_weight', type=str, default='logs/Ant/SAC_DR_test/seed2-20240419-1654/final_model/actor.pth')
    p.add_argument('--trajectory_length', type=int, default=200)
    p.add_argument('--number_of_env', type=int, default=100)
    p.add_argument('--collect_steps', type=int, default=40000)
    args = p.parse_args()
    collect_demo(args)


