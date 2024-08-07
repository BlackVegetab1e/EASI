import os
from time import time, sleep
from datetime import timedelta
from torch.utils.tensorboard import SummaryWriter


class Trainer:

    def __init__(self, env, number_of_env, algo, log_dir, seed=0, num_steps=10**5,
                 eval_interval=10**3, num_eval_episodes=5):
        super().__init__()

        # Env to collect samples.
        self.env = env
        self.number_of_env = number_of_env
        # self.env.seed(seed)


        self.algo = algo
        self.log_dir = log_dir

        # Log setting.
        self.summary_dir = os.path.join(log_dir, 'summary')
        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.model_dir = os.path.join(log_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Other parameters.
        self.num_steps = num_steps
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes

    def train(self):
        # Time to start training.
        self.start_time = time()
        # Episode's timestep.
       
        # Initialize the environment.
        state = self.env.reset()['obs']

        # print(state)
        isaac_gym_sum_step = 0

        while isaac_gym_sum_step < self.num_steps:
            # Pass to the algorithm to update state and episode timestep.
            isaac_gym_sum_step += self.number_of_env
            state, rewards = self.algo.step(self.env, self.number_of_env, state,  isaac_gym_sum_step)

            # Update the algorithm whenever ready.
            if self.algo.is_update(isaac_gym_sum_step):
                for i in range(10):
                    self.algo.update(self.writer)

            # # Evaluate regularly.
            if isaac_gym_sum_step % self.eval_interval == 0:
                pass
                print("isaac_steps:",isaac_gym_sum_step," single_step_reard:", rewards.mean().item())
            #     self.evaluate(step)
                self.algo.save_models(
                    os.path.join(self.model_dir, f'isaac_step{isaac_gym_sum_step}'))

        # Wait for the logging to be finished.
        sleep(10)


    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))



class DR_Trainer:

    def __init__(self, env, number_of_env, algo, log_dir,  sim_params_1, sim_params_2, DR_Type = None, seed=0, num_steps=10**5,
                 eval_interval=10**3, num_eval_steps=20000):
        super().__init__()

        # Env to collect samples.
        self.env = env
        self.number_of_env = number_of_env
        # self.env.seed(seed)


        self.algo = algo
        self.log_dir = log_dir

        # Log setting.
        self.summary_dir = os.path.join(log_dir, 'summary')
        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.model_dir = os.path.join(log_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Other parameters.
        self.num_steps = num_steps
        self.eval_interval = eval_interval
        self.num_eval_steps = num_eval_steps
        self.DR_Type = DR_Type
        self.sim_params_1 = sim_params_1
        self.sim_params_2 = sim_params_2

    def train(self):
        # Time to start training.
        self.start_time = time()
        # Episode's timestep.
       
        # Initialize the environment.
        state = self.env.reset()['obs']

        # print(state)
        isaac_gym_sum_step = 0
        env_steps = [0 for i in range(self.number_of_env)]
        done = [False for i in range(self.number_of_env)]
        while isaac_gym_sum_step < self.num_steps:

            for i in range(self.number_of_env):
                if done[i] or env_steps[i] >= self.env.max_episode_length or env_steps[i] == 0:
                    env_steps[i] = 0
                    #  检测一个回合是否结束，如果回合结束，那么就将参数重新进行采样。
                    if self.DR_Type is not None:
                        if self.DR_Type == 'Gaussian':
                            self.env.sample_params_Gaussian(i, self.sim_params_1, self.sim_params_2)
                        if self.DR_Type == 'Uniform':
                            self.env.sample_params_Uniform(i, self.sim_params_1, self.sim_params_2)
                        if self.DR_Type == 'Origin':
                            pass
                    else:
                        self.env.set_params(i, self.sim_params_1)



            # Pass to the algorithm to update state and episode timestep.
            isaac_gym_sum_step += self.number_of_env
            state, rewards, done = self.algo.step(self.env, self.number_of_env, state,  isaac_gym_sum_step)
            
            for i in range(self.number_of_env):
                env_steps[i] += 1


            # Update the algorithm whenever ready.
            if self.algo.is_update(isaac_gym_sum_step):
                for i in range(10):
                    self.algo.update(self.writer)
                
            self.writer.add_scalar('reward', rewards.mean().item(), isaac_gym_sum_step)

            # # Evaluate regularly.
            if isaac_gym_sum_step % self.eval_interval == 0:
                print("isaac_steps:",isaac_gym_sum_step," single_step_reard:", rewards.mean().item())
                # self.evaluate(isaac_gym_sum_step)
            # TODO 这边添加source domain中的测试调用
                self.algo.save_models(
                    os.path.join(self.model_dir, f'isaac_step{isaac_gym_sum_step}'))

        # Wait for the logging to be finished.
        sleep(10)
    

    # 用于设置参数会导致环境reset的情况
    def train_reset(self):
        # Time to start training.
        self.start_time = time()
        # Episode's timestep.
       
        # Initialize the environment.
        state = self.env.reset()['obs']

        # print(state)
        isaac_gym_sum_step = 0
        resetflag = 0
        env_steps = [0 for i in range(self.number_of_env)]
        done = [False for i in range(self.number_of_env)]
        while isaac_gym_sum_step < self.num_steps:

            if resetflag>=self.env.max_episode_length/2:
                resetflag = 0
                #  检测一个回合是否结束，如果回合结束，那么就将参数重新进行采样。
                if self.DR_Type is not None:
                    if self.DR_Type == 'Gaussian':
                        self.env.sample_params_Gaussian(self.sim_params_1, self.sim_params_2)
                    if self.DR_Type == 'Uniform':
                        self.env.sample_params_Uniform(self.sim_params_1, self.sim_params_2)
                    if self.DR_Type == 'Origin':
                        pass
                else:
                    self.env.set_params(self.sim_params_1)



            # Pass to the algorithm to update state and episode timestep.
            isaac_gym_sum_step += self.number_of_env
            resetflag += 1
            state, rewards, done = self.algo.step(self.env, self.number_of_env, state,  isaac_gym_sum_step)
            
            # Update the algorithm whenever ready.
            if self.algo.is_update(isaac_gym_sum_step):
                for i in range(10):
                    self.algo.update(self.writer)
                
            self.writer.add_scalar('reward', rewards.mean().item(), isaac_gym_sum_step)

            # # Evaluate regularly.
            if isaac_gym_sum_step % self.eval_interval == 0:
                print("isaac_steps:",isaac_gym_sum_step," single_step_reard:", rewards.mean().item())
                # self.evaluate(isaac_gym_sum_step)
            # TODO 这边添加source domain中的测试调用
                self.algo.save_models(
                    os.path.join(self.model_dir, f'isaac_step{isaac_gym_sum_step}'))

        # Wait for the logging to be finished.
        sleep(10)

    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))
    

class FineTune_Trainer:

    def __init__(self, env, target_env_param, number_of_env, algo, log_dir, seed=0, num_steps=10**5,
                 eval_interval=10**3, num_eval_steps=20000, data_budget = 4*10**4):
        super().__init__()

        # Env to collect samples.
        self.env = env
        self.number_of_env = number_of_env
        # self.env.seed(seed)

        self.algo = algo
        self.log_dir = log_dir

        # Log setting.
        self.summary_dir = os.path.join(log_dir, 'summary')
        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.model_dir = os.path.join(log_dir, 'fine_tune_model')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Other parameters.
        self.num_steps = num_steps
        self.eval_interval = eval_interval
        self.num_eval_steps = num_eval_steps
        self.data_budget = data_budget


        env.set_params(params=target_env_param)
    


    def finetune(self):
        # Time to start training.
        self.start_time = time()
        # Episode's timestep.
       
        # Initialize the environment.
        state = self.env.reset()['obs']

        # print(state)
        isaac_gym_sum_step = 0
        env_steps = [0 for i in range(self.number_of_env)]
        done = [False for i in range(self.number_of_env)]
        
        self.algo.save_models(
                    os.path.join(self.model_dir, f'FineTune:{0}'))
        while isaac_gym_sum_step < self.num_steps:

            # Pass to the algorithm to update state and episode timestep.
            isaac_gym_sum_step += self.number_of_env
            
            if isaac_gym_sum_step <= self.data_budget:
                data_budget = False
            else:
                if not data_budget:
                    print('-----------------data full---------------------')
                data_budget = True
                break

            state, rewards, done = self.algo.step(self.env, self.number_of_env, state,  isaac_gym_sum_step,data_budget)
            for i in range(self.number_of_env):
                env_steps[i] += 1


            
                
            self.writer.add_scalar('reward', rewards.mean().item(), isaac_gym_sum_step)

            # # Evaluate regularly.
            if isaac_gym_sum_step % self.eval_interval == 0:
                print("isaac_steps:",isaac_gym_sum_step," single_step_reward:", rewards.mean().item())
                # self.evaluate(isaac_gym_sum_step)

        print('-----------------updating---------------------')
        # Update the algorithm when ready.
        for finetune_iter in range(1,10):
            if self.algo.is_update(isaac_gym_sum_step):
                for i in range(10):
                    self.algo.update(self.writer)
            print(f'update{finetune_iter}times')

            self.algo.save_models(
                    os.path.join(self.model_dir, f'FineTune:{finetune_iter}'))

        # Wait for the logging to be finished.
        sleep(10)


    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))