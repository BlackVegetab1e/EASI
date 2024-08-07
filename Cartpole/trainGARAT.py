
import numpy as np
from isaac_gym_env import paramCartpoleFull
import argparse
from gail_airl_ppo.buffer import ExpertBuffer, RefBufferOfTragectory
from gail_airl_ppo.algo import GARAT, GARATTrainer
import os
import torch

DEVICE = 'cuda:0'


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def run(args):
    setup_seed(args.seed)
    envs = paramCartpoleFull(args.number_of_env, rl_device=DEVICE, seed=args.seed, headless=args.headless)

    params= [0.3, 0.1, 0.3, 3e-04, 2e-03 ,5e-03, 1e-02, 20, 0.3, 5, 0.6]   
    envs.set_params(params)


    state_shape = (*envs.observation_space.shape, 1)
    action_shape = (*envs.action_space.shape, 1)
    buffer_real = ExpertBuffer(expert_path=args.expert_data, device=DEVICE)
    gat = GARAT(buffer_exp=buffer_real, state_shape=state_shape,number_of_envs=args.number_of_env, rollout_length=args.trajectory_length, action_shape= action_shape, seed=args.seed, 
                action_policy_dir = args.expert_weight, device=DEVICE)
    log_dir = os.path.join(
        'logs', args.env_id, args.log_mark, args.tag, f'seed{args.seed}')
    trainer = GARATTrainer(
        env=envs,
        number_of_env = args.number_of_env,
        algo=gat,
        log_dir=log_dir,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        seed=args.seed,
    )
    trainer.train()




if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--env_id', type=str, default='Cart_pole_50hz_paper')
    p.add_argument('--log_mark', type=str, default='GAT_pure')
    p.add_argument('--tag', type=str, default='OOD025')
    
    p.add_argument('--cuda', default=True ,action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--expert_weight', type=str, default='logs/Cart_pole_50hz_paper/SAC_baseline/seed0-20240417-1450/model/isaac_step500000/actor.pth')
    p.add_argument('--expert_data', type=str, default='logs/new_cart_pole_OOD/expert/size40000_traj_length200_real_domain_cpu_seed_0_OOD0.25.pth')
    p.add_argument('--trajectory_length', type=int, default=200)
    p.add_argument('--summary_dir', type=str, default='logs/')
    
    p.add_argument('--number_of_env', type=int, default=100)    
    p.add_argument('--num_steps', type=int, default=3*1e6)   
    p.add_argument('--eval_interval', type=int, default=3*1e5)   

    p.add_argument('--headless', type=bool, default=False)
    args = p.parse_args()
    run(args)

 