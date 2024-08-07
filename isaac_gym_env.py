import isaacgym
import isaacgymenvs
import torch
import numpy as np
from isaacgym import gymapi
from torch.utils.tensorboard import SummaryWriter
import math
import isaacgym
import isaacgymenvs
import torch
import numpy as np
from isaacgym import gymapi
from torch.utils.tensorboard import SummaryWriter


class paramBallBalance():

    def __init__(self, num_envs, rl_device, seed, headless=True, max_episode_length = 500):
        self._envs = isaacgymenvs.make(
            seed=seed, 
            task="BallBalance", 
            num_envs=num_envs, 
            sim_device="cpu",
            rl_device=rl_device,
            graphics_device_id=0,
            headless=headless,
            cfg = None
        )
        self.number_env = num_envs
        self.gym = gymapi.acquire_gym()
        self._envs.max_episode_length = max_episode_length

    

    def reset_isaacgym_env(self ,env_id):
        self._envs.reset_idx(torch.tensor(env_id, device=torch.device('cpu')))
        # envs.reset_idx(torch.tensor(env_id,device=torch.device('cuda')))


    def sample_params_Gaussian(self, params_mean, params_var):
        cov_mat = np.diag(params_var)

        parma_instance = np.empty((self.number_env, len(params_mean)))
        for i in range(self.number_env):
            parma_instance[i] = np.random.multivariate_normal(params_mean,cov_mat)

        self.set_params(parma_instance)
        # print(parma_instance)
        # print('Gaussian')
    
    def sample_params_Uniform(self, lower, upper):
        parma_instance = np.empty((self.number_env, len(lower)))
        for i in range(self.number_env):
            parma_instance[i] = np.random.uniform(lower,upper)
        self.set_params(parma_instance)
    

    def set_params(self, params):
        # 参数设置表：
        # 0  小球质量
        # 1  托盘质量
        # 2  托盘摩擦力
        # 3  托盘反弹系数
        # 4  Actor P
        # 5  Actor D
        # 6  Actor 摩擦力


        params = np.array(params)
        if params.ndim==1:
            all_param = np.empty((self.number_env, len(params)))
            for i in range(self.number_env):
                all_param[i] = params
            params = all_param
            
        assert len(params[0]) == 7

        # print(params[0])
        for env_index in range(self.number_env):
            # 小球质量
            ball_body = self.gym.get_actor_rigid_body_properties(self._envs.envs[env_index], 1)
            # print('0',ball_body[0].mass)
            ball_body[0].mass = params[env_index, 0]
            self.gym.set_actor_rigid_body_properties(self._envs.envs[env_index], 1, ball_body)
        
            # 托盘质量 摩擦力 反弹系数
            tray_body = self.gym.get_actor_rigid_body_properties(self._envs.envs[env_index], 0)
            # print('1',tray_body[0].mass)
            tray_body[0].mass = params[env_index, 1]
            self.gym.set_actor_rigid_body_properties(self._envs.envs[env_index], 0, tray_body)

            tray_shape = self.gym.get_actor_rigid_shape_properties(self._envs.envs[env_index], 0)
            # print('2',tray_shape[0].friction)
            tray_shape[0].friction = params[env_index, 2]
      
            # print('4',tray_shape[0].restitution)
            tray_shape[0].restitution = params[env_index, 3]
            self.gym.set_actor_rigid_shape_properties(self._envs.envs[env_index], 0, tray_shape)
            
            # Actor P D 摩擦力
            # actorDOF:  1 3 5
            property = self.gym.get_actor_dof_properties(self._envs.envs[env_index], 0)
            for i in [1, 3, 5]:
                # print('5',property[i]['stiffness'])
                property[i]['stiffness'] = params[env_index,4]
                # print('6',property[i]['damping'])
                property[i]['damping']  = params[env_index,5]
                # print('7',property[i]['friction'])
                property[i]['friction'] = params[env_index,6]

            self.gym.set_actor_dof_properties(self._envs.envs[env_index], 0, property)




        



    def step(self, action):
        next_state, reward, done, _ = self._envs.step(action)
        if not torch.isnan(next_state['obs']).sum() == 0:
            for i in range(len(next_state['obs'])):
                if not torch.isnan(next_state['obs'][i]).sum() == 0:
                    # print(next_state['obs'][i])
                    # print(done[i])
                    done[i] = True
                    # print(done[i])
                    self._envs.reset_idx([i])
                    next_state['obs'][i] = torch.zeros_like(next_state['obs'][i])

                    # print(next_state['obs'][i])
                    print('环境内部出现一个错误，但是已经自动重置，不会影响结果')
        return next_state, reward, done, _
    




    def reset(self):
        self._envs.reset_idx(torch.arange(self.number_env))
        return self._envs.reset()
    
    @property
    def action_space(self):
        return self._envs.action_space
    
    @property
    def observation_space(self):
        return self._envs.observation_space
    @property
    def max_episode_length(self):
        return self._envs.max_episode_length









if __name__ == "__main__":
    num_envs = 100
    envs = paramBallBalance(num_envs, 'cpu',0,headless=False)

    flag = False
    last_angle = 0
    # 参数设置表：
    # 0  小球质量
    # 1  托盘质量
    # 2  托盘摩擦力

    # 3  托盘反弹系数
    # 4  Actor P
    # 5  Actor D
    # 6  Actor 摩擦力
    # paramas = [1 ,2 ,1 ,0.3 ,4000 ,100 ,1]
    # envs.set_params(paramas)
    # envs.set_params(paramas)
    for __ in range(100):
        obs = envs.reset()
        reward_sum = 0
        mx_speed = 0
        for _ in range(200):
            
            # actions = actor_policy.controller_action(obs['obs'])
            # print(envs.action_space)
            actions = torch.zeros((num_envs, envs.action_space.shape[0]))
            # print(actions)
            obs, reward, done, info = envs.step(actions)
            # time.sleep(0.1)
            reward_sum += reward
            
            print(obs['obs'][0])
    # envs.set_params(None)
        

        

