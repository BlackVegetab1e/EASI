import os
import argparse
from datetime import datetime

import numpy as np
# from gail_airl_ppo.env import make_env
from isaac_gym_env import paramBallBalance
from gail_airl_ppo.algo import SAC
from gail_airl_ppo.trainer import DR_Trainer
import torch


DEVICE = 'cuda:0'



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def run(args):
    setup_seed(args.seed)
    env = paramBallBalance(args.number_of_env, DEVICE, seed=args.seed, headless=True)



    # '''
    # 参数设置表：
    # 0  小球质量
    # 1  托盘质量
    # 2  托盘摩擦力
    # 3  托盘滚动摩擦力
    # 4  托盘反弹系数
    # 5  Actor P
    # 6  Actor D
    # 7  Actor 摩擦力


    Sim_Param = [3,5,1,0.3,100,10,5]

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
    p.add_argument('--num_steps', type=int, default=6*10**6)
    p.add_argument('--eval_interval', type=int, default=3*10**4)
    p.add_argument('--env_id', type=str, default='BallBalance')
    p.add_argument('--cuda', default=True ,action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--number_of_env', type=int, default=200)
    p.add_argument('--log_mark', type=str, default='SAC_DR')
    

    
    args = p.parse_args()
    run(args)
