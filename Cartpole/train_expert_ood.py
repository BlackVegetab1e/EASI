import os
import argparse
from datetime import datetime


# from gail_airl_ppo.env import make_env
from isaac_gym_env import paramCartpoleFull
from gail_airl_ppo.algo import SAC
from gail_airl_ppo.trainer import DR_Trainer
import torch


DEVICE = 'cuda:0'

def run(args):
    env = paramCartpoleFull(args.number_of_env, DEVICE, seed=args.seed, headless=True)

    PRESET_PARMAS = [0.3, 0.1, 0.3, 3e-04, 2e-03 ,5e-03, 1e-02, 20, 0.3, 5, 0.6]
    if args.OOD:
        PRESET_PARMAS[7] *= args.OOD_rate


    algo = SAC(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device(DEVICE if args.cuda else "cpu"),
        seed=args.seed,
        number_of_envs= args.number_of_env
    )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    if args.OOD:
        log_dir = os.path.join(
            'logs', args.env_id, 'SAC_baseline', f'seed{args.seed}-{time}-OOD{args.OOD_rate}')
    else:
        log_dir = os.path.join(
            'logs', args.env_id, 'SAC_baseline', f'seed{args.seed}-{time}')

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
    p.add_argument('--num_steps', type=int, default=5*10**5)
    p.add_argument('--eval_interval', type=int, default=5*10**3)
    p.add_argument('--env_id', type=str, default='Cart_pole_50hz_paper')
    p.add_argument('--cuda', default=True ,action='store_true')
    p.add_argument('--seed', type=int, default=2)
    p.add_argument('--number_of_env', type=int, default=100)
    p.add_argument('--OOD', action='store_true', default=False)
    p.add_argument('--OOD_rate', type=float, default=1)

    
    args = p.parse_args()
    run(args)
