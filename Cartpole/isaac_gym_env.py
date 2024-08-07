import isaacgym
import isaacgymenvs
import torch
import numpy as np
from isaacgym import gymapi
from torch.utils.tensorboard import SummaryWriter
import math

    
class paramCartpoleFull():
    
    def __init__(self, num_envs, rl_device, seed, headless=True, max_episode_length = 200):
        self._envs = isaacgymenvs.make(
            seed = seed, 
            task="CartpoleRealFull", 
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
        self.Angle_Offset = None
        self.Pose_Limit  = None
        self.rl_device = rl_device
        self.action_range = [[-0.2,0.2]]

    def reset_isaacgym_env(self ,env_id):
        self._envs.reset_idx(torch.tensor(env_id, device=torch.device('cpu')))
       

    def sample_params_Gaussian(self, params_mean, params_var):
        # print(params_mean)
        # print(params_var)
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


    def set_params(self, params, set_length = True):
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

        '''
        body :  {'cart': 1, 'pole': 2, 'slider': 0}
        DOF  :  {'cart_to_pole': 1, 'slider_to_cart': 0}
        '''
        params = np.array(params)
        if params.ndim==1:
            all_param = np.empty((self.number_env, len(params)))
            for i in range(self.number_env):
                all_param[i] = params
            params = all_param
            


        assert len(params[0]) == 11


        # Param0 : Pole Length
        if set_length:
            self._envs.reset_sim_with_param(params[:,0])
        

        for env_index in range(self.number_env):
            property = self.gym.get_actor_rigid_body_properties(self._envs.envs[env_index], 0)
            # Param1 : Pole Mass
            property[2].mass  = params[env_index, 1]
            # Param2 : Cart Mass
            property[1].mass  = params[env_index, 2]

            self.gym.set_actor_rigid_body_properties(self._envs.envs[env_index], 0, property, recomputeInertia=True)
            property = self.gym.get_actor_dof_properties(self._envs.envs[env_index], 0)
            # Param3 : Pole DOF_Friction
            property[1]['friction'] = params[env_index, 3]
            # Param4 : Pole DOF_Dump
            property[1]['damping']  = params[env_index, 4]
            # Param5 : Pole Dof_amature
            property[1]['armature'] = params[env_index, 5]
            # Param6 : Cart DOF_Friction
            property[0]['friction'] = params[env_index, 6]
            # Param7 : Cart PID_P
            property[0]['stiffness']  = params[env_index, 7]
            # Param8 : Cart PID_I
            property[0]['damping']    = params[env_index, 8]
            # Param9 Cart EffortLimit      
            property[0]['effort']    = params[env_index, 9]
            # Param10 Cart EffortLimit      
            property[0]['velocity']    = params[env_index, 10]
            self.gym.set_actor_dof_properties(self._envs.envs[env_index], 0, property)




        # for env_index in range(self.number_env):
            
        #     property = self.gym.get_actor_rigid_body_properties(self._envs.envs[env_index], 0)
        #     property1 = self.gym.get_actor_dof_properties(self._envs.envs[env_index], 0)
        #     print('mass:',property[2].mass,'com:',property[2].com.z,\
        #           'friction:',property1[0]['friction'],'damping:',property1[0]['damping'] ,'armature:',property1[0]['armature'] )
        #     print('##############')



    def step(self, action):
        # print(action)
        # if self.Pose_Limit is not None:
        #     action = torch.where(action>self.Pose_Limit,   self.Pose_Limit, action)
        #     action = torch.where(action<-self.Pose_Limit, -self.Pose_Limit, action)


        obs, reward, done, info = self._envs.step(action)
        # print(obs['obs'])
        # print(obs['obs'][:,2])
        # print(self.Angle_Offset)
        # if self.Angle_Offset is not None:
        #     obs['obs'][:,2] += self.Angle_Offset
        # print(obs['obs'])
        return obs, reward, done, info
    

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
    num_envs = 1
    envs = paramCartpoleFull(num_envs, 'cpu',0,headless=False)
    # envs.get_mass(0)
    # envs.get_contact(0)
    import time
    # envs.set_params(2,[0.3,3,1,0,0])
    # envs.set_params(0,[0.3,3,1,0,0])
    '''
        ID    Param
        0     Pole Length
        1     Pole Mass
        2     Cart Mass
        3     Pole DOF_Friction
        4     Pole DOF_Dump
        5     Pole Dof_amature
        6     Cart DOF_Friction
        7     Cart PID_P
        8     Cart PID_D
    '''
    # envs.set_params(np.array([[0.2, 0.03, 1, 0.001, 0.01, 0.005, 0.1, 10, 5]]), set_length = True)
    from gail_airl_ppo.algo import SACExpert
    actor_policy = SACExpert(
        state_shape=(4,1),
        action_shape=(1,1),
        device=torch.device('cpu'),
        # path='logs/Cartpole/SAC_DR_OA_Mid/seed0-20240325-1635/model/isaac_step1000000/actor.pth')
        path='logs/new_cart_pole/SAC_DR/seed0-20240515-1427/final_model/actor.pth')

    params= [0.3, 0.1, 0.3, 3e-04, 2e-03 ,5e-03, 1e-02, 20, 0.3, 5, 0.6]

    envs.set_params(params)
    flag = False
    last_angle = 0
    for __ in range(1000000):
        obs = envs.reset()
        reward_sum = 0
        mx_speed = 0
        for _ in range(200):
            
            actions = actor_policy.controller_action(obs['obs'])

            print(actions)
            obs, reward, done, info = envs.step(actions)
            # time.sleep(0.1)
            reward_sum += reward
            print(obs['obs'])
        

        

