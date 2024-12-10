import os
import argparse
from datetime import datetime
import numpy as np

# from gail_airl_ppo.env import make_env
from isaac_gym_env import paramAnt, paramBallBalance, paramCartpoleFull
from torch.utils.tensorboard import SummaryWriter
from EvolutionaryAdversarial.algo import StateTransDiscriminator, SACExpert
from EvolutionaryAdversarial.algo import Evo_Gaussian
from EvolutionaryAdversarial.buffer import ExpertBuffer, RefBufferOfTragectory
import torch


# training device
DEVICE = 'cuda:0'


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def run(args):
    # 设置种子
    setup_seed(args.seed)

    if args.env_id == 'Ant': 
        envs = paramAnt(args.number_of_env, rl_device=DEVICE, seed=args.seed, headless=args.headless)
        Sim_Param = [1.5, 0.3, 0.2, 0.3, 0.1,  0.1, 0.2, 0.1, 0.1, 0.2, 1]    
        SIM_PARAMS_Lower = [0.33*i for i in Sim_Param] 
        SIM_PARAMS_Upper = [3*i for i in Sim_Param]
        if args.OOD:
            SIM_PARAMS_Lower[0] = 0*Sim_Param[0]
            SIM_PARAMS_Lower[2] = 3*Sim_Param[2]
            SIM_PARAMS_Lower[10] = 0*Sim_Param[10]
            SIM_PARAMS_Upper[0] = Sim_Param[0]
            SIM_PARAMS_Upper[2] = 5*Sim_Param[2]
            SIM_PARAMS_Upper[10] = Sim_Param[10]
        DNA_BOUND_single = {0:[0  ,  0],
                            1:[0  ,  0],
                            2:[0  ,  0],
                            3:[0  ,  0],
                            4:[0  ,  0],
                            5:[0  ,  0],
                            6:[0  ,  0],
                            7:[0  ,  0],
                            8:[0  ,  0],
                            9:[0  ,  0],
                            10:[0  ,  0]}
        DNA_SIZE = len(DNA_BOUND_single)
        for i in range(DNA_SIZE):
            DNA_BOUND_single[i][0] = SIM_PARAMS_Lower[i]
            DNA_BOUND_single[i][1] = SIM_PARAMS_Upper[i]

        need_clip = [True for i in range(DNA_SIZE)]

        if args.OOD:
            need_clip[10] = False
            print('param_10 will not clip')
            need_clip[0] = False
            print('param_'+str(0)+' will not clip')
            need_clip[2] = False
            print('param_'+str(2)+' will not clip')
            need_clip[10] = False
            print('param_'+str(10)+' will not clip')

    elif args.task_env == 'Cartpole': 
        envs = paramCartpoleFull(args.number_of_env, rl_device=DEVICE, seed=args.seed, headless=args.headless)
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


    elif args.task_env == 'Ballbalance': 
        envs = paramBallBalance(args.number_of_env, rl_device=DEVICE, seed=args.seed, headless=args.headless)
        Sim_Param = [3,5,1,0.3,100,10,5]

        SIM_PARAMS_Lower = [0.33*i for i in Sim_Param] 
        SIM_PARAMS_Upper = [3*i for i in Sim_Param]

        DNA_BOUND_single = {0:[0  ,  0],
                            1:[0  ,  0],
                            2:[0  ,  0],
                            3:[0  ,  0],
                            4:[0  ,  0],
                            5:[0  ,  0],
                            6:[0  ,  0]}
        DNA_SIZE = len(DNA_BOUND_single)
        for i in range(DNA_SIZE):
            DNA_BOUND_single[i][0] = SIM_PARAMS_Lower[i]
            DNA_BOUND_single[i][1] = SIM_PARAMS_Upper[i]
        
        need_clip = [True for i in range(DNA_SIZE)]
        if(args.OOD):
            need_clip[0] = False
    
    else:
        print('Wrong task name')
        return


    POP_SIZE = args.number_of_env

    evo = Evo_Gaussian(DNA_SIZE = DNA_SIZE,DNA_BOUND = DNA_BOUND_single, \
                      POP_SIZE = POP_SIZE, SURVIVE_RATE = args.survive_rate, need_clip= need_clip)
    param_evalution = np.zeros((50,300,11))


    state_shape = (*envs.observation_space.shape, 1)
    action_shape = (*envs.action_space.shape, 1)


    buffer_real = ExpertBuffer(expert_path=args.expert_data, device=DEVICE)
    expert_policy = SACExpert(
        state_shape=state_shape,
        action_shape=action_shape,
        device=torch.device(DEVICE),
        path=args.expert_weight)


    state_disc = StateTransDiscriminator(
    buffer_exp=buffer_real,
    state_shape=state_shape,
    action_shape=action_shape,
    device=torch.device(DEVICE),
    batch_size = args.disc_batch_size_per_env*args.number_of_env,
    units_disc=(128, 128),
    epoch_disc=20
    )


    search_logdir = args.summary_dir+'/'+args.env_id+'/search_gaussian/seed_'+str(args.seed)+args.tag+'/'
    writer = SummaryWriter(log_dir= search_logdir)
    
    ref_tragectory_buffer = RefBufferOfTragectory(args.number_of_env, torch.device(DEVICE))
      

    for step_numbers in range(50):
        params_pop = evo.make_kids_Gaussian()
        if (step_numbers+1)%10 == 0:
            evo.show_plot_mean_var(step_numbers+1, search_logdir)
        for i in range(DNA_SIZE):
            writer.add_histogram('param/pop'+str(i),params_pop[:,i], step_numbers)
        
        param_evalution[step_numbers] = params_pop

        tragectory_pop = state_disc.step(envs, expert_policy=expert_policy, \
                                   step_length = args.trajectory_length, \
                                    params=params_pop)

        for i in range(len(tragectory_pop)):
            ref_tragectory_buffer.append(tragectory_pop[i])

        state_disc.update(writer, ref_tragectory_buffer)
        reward = state_disc.disc.calculate_reward_WGail(tragectory_pop).cpu().numpy()
        evo.select_elite(reward)   # keep some good parent for elitism
    for gen in range(50):
        np.savetxt(search_logdir+"all_process"+str(gen)+".csv", param_evalution[gen], delimiter="," )



if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--num_steps', type=int, default=10**6)
    p.add_argument('--eval_interval', type=int, default=10**4)
    p.add_argument('--env_id', type=str, default='Ant')
    p.add_argument('--cuda', default=True ,action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--expert_weight', type=str, default='example/example_policy/Ant_DR/actor.pth')
    p.add_argument('--expert_data', type=str, default='example/example_expert_state_trans/Ant_DR/size40000_traj_length200_real_domain_cpu_seed_0.pth')
    p.add_argument('--trajectory_length', type=int, default=200)
    p.add_argument('--summary_dir', type=str, default='logs/')

    p.add_argument('--tag', type=str, default='')
    p.add_argument('--OOD', action='store_true', default=False)

    p.add_argument('--number_of_env', type=int, default=300)
    p.add_argument('--survive_rate', type=float, default=0.5)
    p.add_argument('--disc_batch_size_per_env', type=int, default=16)
    p.add_argument('--headless', type=bool, default=True)
    args = p.parse_args()
    run(args)



    