import os
import argparse
from datetime import datetime
import numpy as np

# from gail_airl_ppo.env import make_env
from isaac_gym_env import paramCartpoleFull
from torch.utils.tensorboard import SummaryWriter
from gail_airl_ppo.algo import GAIL, SACExpert
from gail_airl_ppo.trainer import Trainer
from gail_airl_ppo.buffer import ExpertBuffer, RefBufferOfTragectory
import torch
from Evolution_Gaussian import Evo_Gaussian


DEVICE = 'cuda:0'


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)



def run(args):
    # 设置种子
    setup_seed(args.seed)
    envs = paramCartpoleFull(args.number_of_env, rl_device=DEVICE, seed=args.seed, headless=args.headless)
    state_shape = (*envs.observation_space.shape, 1)
    action_shape = (*envs.action_space.shape, 1)
    buffer_real = ExpertBuffer(expert_path=args.expert_data, device=DEVICE)

    

    gail = GAIL(
    buffer_exp=buffer_real,
    state_shape=state_shape,
    action_shape=action_shape,

    tragectorch_length= args.trajectory_length, 
    device=torch.device(DEVICE),
    seed=args.seed,
    batch_size = args.disc_batch_size_per_env*args.number_of_env,

    units_disc=(128, 128),
    epoch_ppo=2,
    epoch_disc=10

    )

    # action = algo.exploit(state)
    actor_policy = SACExpert(
        state_shape=state_shape,
        action_shape=action_shape,
        device=torch.device(DEVICE),
        path=args.expert_weight)
    train_env_policy(gail, envs, actor_policy, args)


def train_env_policy(gail:GAIL, envs, expert_policy, args):

    number_of_env = args.number_of_env
    if args.OOD:
        search_logdir = args.summary_dir+'/'+args.env_id+'/search_gaussian/'+args.tag+'seed_'+str(args.seed)+'OOD'+str(args.OOD_rate)+'/'
    else:
        search_logdir = args.summary_dir+'/'+args.env_id+'/search_gaussian/'+args.tag+'seed_'+str(args.seed)+'/'
    writer = SummaryWriter(log_dir= search_logdir)
    
    ref_tragectory_buffer = RefBufferOfTragectory(args.number_of_env, torch.device(DEVICE))
    
    # DNA_BOUND_single = {0:[0.1, 0.5],
    #                     1:[0, 1],
    #                     2:[0, 1],
    #                     3:[0.0, 1],
    #                     4:[0.0, 0.5]}

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
        10    Cart Vel
    '''
    DNA_BOUND_single = {0:[0.0, 0.0],
                        1:[0.00, 0.0],
                        2:[0.00, 0],
                        3:[0.0, 0.0],
                        4:[0.0, 0.0],
                        5:[0.0, 0.0],
                        6:[0.0, 0.],
                        7:[0, 0],
                        8:[0, 0],
                        9:[0,0],
                        10:[0,0]}
    DNA_SIZE = len(DNA_BOUND_single)

    POP_SIZE = args.number_of_env

    PRESET_PARMAS = [0.3, 0.1, 0.3, 3e-04, 2e-03 ,5e-03, 1e-02, 20, 0.3, 5, 0.6]

    for i in range(DNA_SIZE):
        DNA_BOUND_single[i][0] = PRESET_PARMAS[i] / 3 
        DNA_BOUND_single[i][1] = PRESET_PARMAS[i] * 3 
    
    need_clip = [True for i in range(DNA_SIZE)]
    need_clip[7] = False

    evo = Evo_Gaussian(DNA_SIZE = DNA_SIZE,DNA_BOUND = DNA_BOUND_single, \
                      POP_SIZE = POP_SIZE, SURVIVE_RATE = args.survive_rate, need_clip = need_clip)

    for step_numbers in range(50):
        params_pop = evo.make_kids_Gaussian()
        if (step_numbers+1)%10 == 0:
            evo.show_plot_mean_var(step_numbers+1, search_logdir)
        for i in range(DNA_SIZE):
            writer.add_histogram('param/pop'+str(i),params_pop[:,i], step_numbers)

        tragectory_pop = gail.step(envs, expert_policy=expert_policy, \
                                   step_length = args.trajectory_length, \
                                    params=params_pop)

        
        for i in range(len(tragectory_pop)):
            ref_tragectory_buffer.append(tragectory_pop[i])

        gail.update(writer, ref_tragectory_buffer)
        reward = gail.disc.calculate_reward_WGail(tragectory_pop).cpu().numpy()
        evo.select_elite(reward)   # keep some good parent for elitism



 
    
 

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--num_steps', type=int, default=10**6)
    p.add_argument('--eval_interval', type=int, default=10**4)
    p.add_argument('--env_id', type=str, default='Cart_pole_50hz_paper')
    p.add_argument('--cuda', default=True ,action='store_true')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--expert_weight', type=str, default='logs/Cart_pole_50hz_paper/SAC_DR/seed1-20240416-1946/final_model/actor.pth')
    p.add_argument('--expert_data', type=str, default='logs/Cart_pole_50hz_paper/expert/size40000_traj_length200_real_domain_cpu_seed_1.pth')
    p.add_argument('--trajectory_length', type=int, default=200)
    p.add_argument('--summary_dir', type=str, default='logs/')

    p.add_argument('--number_of_env', type=int, default=300)
    p.add_argument('--survive_rate', type=float, default=0.5)
    p.add_argument('--disc_batch_size_per_env', type=int, default=16)
    p.add_argument('--headless', type=bool, default=True)

    p.add_argument('--tag', type=str, default='')

    p.add_argument('--OOD', action='store_true', default=False)
    p.add_argument('--OOD_rate', type=float, default=1)

    args = p.parse_args()
    run(args)



    