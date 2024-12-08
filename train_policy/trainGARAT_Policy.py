import os
import argparse
from datetime import datetime

import numpy as np
# from gail_airl_ppo.env import make_env
from isaac_gym_env import paramAnt
from EvolutionaryAdversarial.algo import SAC
from EvolutionaryAdversarial.trainer import DR_Trainer
import torch
from EvolutionaryAdversarial.algo import GARAT

DEVICE = 'cuda:0'


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)




def run(args):
    setup_seed(args.seed)
    env = paramAnt(args.number_of_env, DEVICE, seed=args.seed, headless=True)

    PRESET_PARMAS =  [1.5, 0.3,   0.2, 0.3, 0.1,   0.1, 0.2, 0.1,  0.1, 0.2, 1]    

    algo = SAC(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device(DEVICE if args.cuda else "cpu"),
        seed=args.seed,
        number_of_envs= args.number_of_env
    )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, 'SAC_baseline', str(args.OOD_RATE), f'seed{args.seed}-{time}')
    
    GAT_ANT = GARAT(buffer_exp=None, state_shape=env.observation_space.shape, 
                    action_shape=env.action_space.shape, number_of_envs=1, 
                    device=DEVICE,seed=args.seed, action_policy_dir = None )
    GAT_ANT.load_actor(args.GAT_policy)

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
        sim_params_2=None,
        GAT_policy = GAT_ANT
    )
    trainer.train_reset()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--num_steps', type=int, default=1*10**7)
    p.add_argument('--eval_interval', type=int, default=10**5)
    p.add_argument('--env_id', type=str, default='Ant_GAT_pure')
    p.add_argument('--cuda', default=True ,action='store_true')
    p.add_argument('--seed', type=int, default=2)
    p.add_argument('--number_of_env', type=int, default=200)
    p.add_argument('--OOD_RATE',type=float, default=0.25)

    p.add_argument('--GAT_policy', type=str, default='logs/Ant_GAT/GAT_OOD/OOD025/seed0/model/isaac_step20000000/actor.pth')
    

    
    args = p.parse_args()
    run(args)
