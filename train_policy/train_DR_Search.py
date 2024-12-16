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
from EvolutionaryAdversarial.algo import SAC
from EvolutionaryAdversarial.trainer import DR_Trainer
import torch


DEVICE = 'cuda:0'

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def run(args):
    setup_seed(args.seed)
    if args.env_id == 'Ant': 
        env = paramAnt(args.number_of_env, DEVICE, seed=args.seed, headless=True)

    elif args.env_id == 'Ballbalance':
        env = paramBallBalance(args.number_of_env, DEVICE, seed=args.seed, headless=True) 

    elif args.env_id == 'Cartpole':
        env = paramCartpoleFull(args.number_of_env, DEVICE, seed=args.seed, headless=True)  
    else:
        print("WRONG NAME")
        return

    SIM_PARAMS_MEAN = np.loadtxt(args.search_params_dir+'/50_mean.csv')
    SIM_PARAMS_VAR =  np.loadtxt(args.search_params_dir+'/50_var.csv')              


    # SIM_PARAMS_VAR = [2*i for i in SIM_PARAMS_VAR]


    algo = SAC(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device(DEVICE if args.cuda else "cpu"),
        seed=args.seed,
        number_of_envs= args.number_of_env
    )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, args.log_mark, args.tag, f'seed{args.seed}-{time}')

    
    trainer = DR_Trainer(
        env=env,
        number_of_env = args.number_of_env,
        algo=algo,
        log_dir=log_dir,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        seed=args.seed,
        DR_Type ='Gaussian',
        sim_params_1=SIM_PARAMS_MEAN,
        sim_params_2=SIM_PARAMS_VAR

    )
    trainer.train_reset()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--num_steps', type=int, default=1*10**7, help='训练多少步(总的环境step)')
    p.add_argument('--eval_interval', type=int, default=10**5, help='多久记录一下当前的训练表现')
    p.add_argument('--env_id', type=str, default='Ant', help='这是什么环境')
    p.add_argument('--log_mark', type=str, default='SAC_DR_search', help='标记一下这个实验是干啥的')
    p.add_argument('--cuda', default=True ,action='store_true', help='用哪个设备')
    p.add_argument('--seed', type=int, default=0, help='随机种子')
    p.add_argument('--number_of_env', type=int, default=200, help='多少个并行环境')
    
    p.add_argument('--tag', type=str, default='', help='标记一下这个实验是干啥的(有时候一个标不够用)')
    p.add_argument('--search_params_dir', type=str, default='logs/Ant/search_gaussian/seed_0', help='EASI搜索的参数存在哪里了,直接把EASI的结果路径复制过来就行')

    
    args = p.parse_args()
    print(args.num_steps)
    run(args)
