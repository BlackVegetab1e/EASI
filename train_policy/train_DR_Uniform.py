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
from isaac_gym_env import paramAnt, paramBallBalance, paramCartpoleFull
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
        Sim_Param = [1.5, 0.3,   0.2, 0.3, 0.1,   0.1, 0.2, 0.1,  0.1, 0.2, 1]     
    elif args.env_id == 'Ballbalance':
        env = paramBallBalance(args.number_of_env, DEVICE, seed=args.seed, headless=True) 
        Sim_Param = [3, 5, 1, 0.3 ,100 ,10 ,5]
    elif args.env_id == 'Cartpole':
        env = paramCartpoleFull(args.number_of_env, DEVICE, seed=args.seed, headless=True)  
        Sim_Param = [0.3, 0.1, 0.3, 3e-04, 2e-03 ,5e-03, 1e-02, 20, 0.3, 5, 0.6]
    else:
        print("WRONG NAME")
        return

    SIM_PARAMS_Lower = [0.33*i for i in Sim_Param] 
    SIM_PARAMS_Upper = [3*i for i in Sim_Param]
    
    env.set_params(SIM_PARAMS_Lower)



    algo = SAC(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device(DEVICE if args.cuda else "cpu"),
        seed=args.seed,
        number_of_envs= args.number_of_env
    )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, args.log_mark, f'seed{args.seed}-{time}')
    
    trainer = DR_Trainer(
        env=env,
        number_of_env = args.number_of_env,
        algo=algo,
        log_dir=log_dir,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        seed=args.seed,
        DR_Type ='Uniform',
        sim_params_1=SIM_PARAMS_Lower,
        sim_params_2=SIM_PARAMS_Upper

    )
    trainer.train_reset()
    algo.save_all_models(os.path.join(log_dir, 'final_model'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--num_steps', type=int, default=1*10**7, help='训练多少步(总的环境step)')
    p.add_argument('--eval_interval', type=int, default=10**5, help='多久记录一下当前的训练表现')
    p.add_argument('--env_id', type=str, default='Ant', help='这是什么环境')
    p.add_argument('--log_mark', type=str, default='SAC_DR', help='标记一下这个实验是干啥的')
    p.add_argument('--cuda', default=True ,action='store_true', help='用哪个设备')
    p.add_argument('--seed', type=int, default=0, help='随机种子')
    p.add_argument('--number_of_env', type=int, default=200, help='多少个并行环境')
    
    args = p.parse_args()
    run(args)
