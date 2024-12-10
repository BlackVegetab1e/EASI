import os
import argparse
from datetime import datetime


# from gail_airl_ppo.env import make_env
from isaac_gym_env import paramAnt
from EvolutionaryAdversarial.algo import SAC
from EvolutionaryAdversarial.trainer import DR_Trainer
import torch


DEVICE = 'cuda:0'


# TODO 这里只需要用一个Ant作为例子即可
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def run(args):
    setup_seed(args.seed)
    env = paramAnt(args.number_of_env, DEVICE, seed=args.seed, headless=True)



        # '''
    # 参数设置表：
    # index|         param                |   value  |    DR_min | DR_max
    #    0        foot friction            (1+0)/2=0.5       0      3
    #    1        foot restitution          0.3              0      1
    #    2        body-leg-DOF friction     0.1              0      0.4
    #    3        body-leg-DOF damping      0.3              0      1
    #    4        body-leg-DOF armature     0.05             0      0.2
    #    5        foot-leg-DOF friction     0.05             0      0.2
    #    6        foot-leg-DOF damping      0.2              0      1
    #    7        foot-leg-DOF armature     0.02             0      0.2
    #    8        foot mass                 0.02             0      0.2
    #    9        leg mass                  0.08             0      0.2
    #    10       body mass                 0.6              0      2
    # '''


    Sim_Param = [1.5, 0.3,   0.2, 0.3, 0.1,   0.1, 0.2, 0.1,  0.1, 0.2, 1]        

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
        'logs', args.env_id, args.log_mark, str(args.OOD_rate) , f'seed{args.seed}-{time}')
    
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

    true_param = Sim_Param 
    if args.OOD:
        true_param[10] *= args.OOD_rate
    trainer.fine_tune(true_param,40000)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--num_steps', type=int, default=1*10**7)
    p.add_argument('--eval_interval', type=int, default=10**5)
    p.add_argument('--env_id', type=str, default='Ant')
    p.add_argument('--cuda', default=True ,action='store_true')
    p.add_argument('--seed', type=int, default=2)
    p.add_argument('--number_of_env', type=int, default=100)
    p.add_argument('--log_mark', type=str, default='SAC_DR_FT_new')
    
    p.add_argument('--OOD', action='store_true', default=True)
    p.add_argument('--OOD_rate', type=float, default=0.5)
    
    args = p.parse_args()
    run(args)
