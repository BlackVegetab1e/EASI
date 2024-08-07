import os
import argparse
from datetime import datetime


# from gail_airl_ppo.env import make_env
from isaac_gym_env import paramCartpoleFull
from gail_airl_ppo.algo import SAC
from gail_airl_ppo.trainer import FineTune_Trainer
import torch
import numpy as np


DEVICE = 'cuda:0'


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)





def run(args):
    setup_seed(args.seed)
    PRESET_PARMAS = [0.3, 0.1, 0.3, 3e-04, 2e-03 ,5e-03, 1e-02, 20, 0.3, 5, 0.6]
    '''
            ID    Param
            0     Pole Length            0.2  -- 0.5
            1     Pole Mass              0.03 -- 0.3
            2     Cart Mass              0.1  -- 1
            3     Pole DOF_Friction      0    -- 0.001
            4     Pole DOF_Dump          0    -- 0.01
            5     Pole Dof_amature       0    -- 0.005
            6     Cart DOF_Friction      0    -- 0.1
            7     Cart PID_P             10   -- 100
            8     Cart PID_D             0    -- 5
            9     Cart EffortLimit       0    -- 10
            10    Cart Vel               0.2  -- 1
    '''
    env = paramCartpoleFull(args.number_of_env, DEVICE, seed=args.seed, headless=True)
    
    # for i in range(env.number_env):
    #     # print(params[i])
    #     env.set_params(env_index = i, params=PRESET_PARMAS)
    

    algo = SAC(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device(DEVICE if args.cuda else "cpu"),
        seed=args.seed,
        number_of_envs= args.number_of_env,
        start_steps=0,   # 已经是训练好的模型，不需要最刚开始进行随机采样
        FT_Mode = True
    )



    # 导入待微调的策略

    algo.fine_tune_load(args.model_dir)

   


    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, 'SAC_DR_FT', f'seed{args.seed}-{time}')
    
    trainer = FineTune_Trainer(
        env=env,
        target_env_param=PRESET_PARMAS,
        number_of_env = args.number_of_env,
        algo=algo,
        log_dir=log_dir,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        seed=args.seed,
        data_budget=args.data_budget
    )
    trainer.finetune()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--num_steps', type=int, default=3*10**5)
    p.add_argument('--eval_interval', type=int, default=10**4)
    p.add_argument('--env_id', type=str, default='Cart_pole_50hz_paper')
    p.add_argument('--model_dir', type=str, default='logs/Cart_pole_50hz_paper/SAC_DR/seed2-20240416-1949/final_model/')
    p.add_argument('--cuda', default=True ,action='store_true')
    p.add_argument('--seed', type=int, default=2)
    p.add_argument('--number_of_env', type=int, default=100)
    p.add_argument('--data_budget', type=int, default=40000)
    

    
    args = p.parse_args()
    run(args)
