import os
import argparse
from datetime import datetime


# from gail_airl_ppo.env import make_env
from isaac_gym_env import paramAnt
from gail_airl_ppo.algo import SAC
from gail_airl_ppo.trainer import DR_Trainer
import torch


DEVICE = 'cuda:0'

def run(args):
    print(args.seed)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--num_steps', type=int, default=1*10**7)
    p.add_argument('--eval_interval', type=int, default=10**5)
    p.add_argument('--env_id', type=str, default='Ant')
    p.add_argument('--cuda', default=True ,action='store_true')
    p.add_argument('--seed', type=int, default=2)
    p.add_argument('--number_of_env', type=int, default=200)
    p.add_argument('--log_mark', type=str, default='SAC_DR')
    

    
    args = p.parse_args()
    run(args)

