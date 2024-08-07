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
    '''
        ID    Param
        0     Pole Length            0.2  -- 0.5
        1     Pole Mass              0.01 -- 0.3
        2     Cart Mass              0.01 -- 1
        3     Pole DOF_Friction      0    -- 0.001
        4     Pole DOF_Dump          0    -- 0.01
        5     Pole Dof_amature       0    -- 0.005
        6     Cart DOF_Friction      0    -- 0.1
        7     Cart PID_P             10   -- 100
        8     Cart PID_D             0    -- 5
        9     Cart EffortLimit       1    -- 10
        10    Cart Vel               0.2  -- 1
    '''
    PARAM = [0.3, 0.1, 0.3, 3e-04, 2e-03 ,5e-03, 1e-02, 20, 0.3, 5, 0.6]
    SIM_PARAMS_Lower = [i/3 for i in PARAM]
    SIM_PARAMS_Upper = [i*3 for i in PARAM]
    
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
        'logs', args.env_id, args.log_mark, str(args.OOD_rate), f'seed{args.seed}-{time}')
    
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


    true_param = PARAM 
    if args.OOD:
        true_param[7] *= args.OOD_rate
    trainer.fine_tune(true_param,40000)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--num_steps', type=int, default=5*10**5)
    p.add_argument('--eval_interval', type=int, default=5*10**3)
    p.add_argument('--env_id', type=str, default='Cart_pole_50hz_paper')
    p.add_argument('--cuda', default=True ,action='store_true')
    p.add_argument('--seed', type=int, default=2)
    p.add_argument('--number_of_env', type=int, default=100)
    p.add_argument('--log_mark', type=str, default='SAC_DR_FT')
    
    p.add_argument('--OOD', action='store_true', default=True)
    p.add_argument('--OOD_rate', type=float, default=0.5)
    
    args = p.parse_args()
    run(args)
