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
    env = paramBallBalance(args.number_of_env, headless=True, seed=args.seed,rl_device=DEVICE)

    PRESET_PARMAS =  [3,5,1,0.3,100,10,5]
    if(args.OOD):
        PRESET_PARMAS[0] *= args.OOD_rate
    algo = SAC(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device(DEVICE if args.cuda else "cpu"),
        seed=args.seed,
        number_of_envs= args.number_of_env
    )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, 'SAC_baseline', f'seed{args.seed}-{time}-OOD{args.OOD_rate}')

    trainer = DR_Trainer(
        env=env,
        number_of_env = args.number_of_env,
        algo=algo,
        log_dir=log_dir,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        seed=args.seed,
        DR_Type =None,
        sim_params_1=PRESET_PARMAS,
        sim_params_2=None

    )
    trainer.train_reset()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--num_steps', type=int, default=6*10**6)
    p.add_argument('--eval_interval', type=int, default=3*10**4)
    p.add_argument('--env_id', type=str, default='BallBalance')
    p.add_argument('--cuda', default=True ,action='store_true')
    p.add_argument('--seed', type=int, default=3)
    p.add_argument('--number_of_env', type=int, default=500)
    
    
    p.add_argument('--OOD', type=bool, default=True)
    p.add_argument('--OOD_rate', type=float, default=1)
    args = p.parse_args()

    if(args.OOD):
        args.env_id = args.env_id + '_OOD'
    run(args)
