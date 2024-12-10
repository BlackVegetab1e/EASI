from IsaacGymEnvs import isaacgymenvs
# import isaacgymenvs
import numpy as np
from isaacgym import gymapi
import torch


class paramAnt():
    # 先进行参数的切分，
    # 参数表:
    # 0    所有脚与地面friction 
    # 1    所有脚与地面restitution
    # 234  身体-腿电机friction damping armature     
    # 567  脚-腿电机friction damping armature
    # 8    脚mass
    # 9    腿mass
    # 10   身体mass 
    def __init__(self, num_envs, rl_device, seed, headless=True, max_episode_length = 200):
        self._envs = isaacgymenvs.make(
            seed=seed, 
            task="Ant", 
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
        # 先进行参数的切分，
        # 参数表:
        # 0    所有脚与地面friction 
        # 1    所有脚与地面restitution
        # 234  身体-腿电机friction damping armature     
        # 567  脚-腿电机friction damping amarmature
        # 8    脚mass
        # 9    腿mass
        # 10   身体mass  

        params = np.array(params)
        if params.ndim==1:
            all_param = np.empty((self.number_env, len(params)))
            for i in range(self.number_env):
                all_param[i] = params
            params = all_param
            


        assert len(params[0]) == 11

        contact_param = params[:, 0:2]
        DOF_param = params[:, 2:8]
        mass_param = params[:, 8:11]
        self.set_contact( contact_param)
        self.set_DOF(DOF_param)
        self.set_mass(mass_param)

    def set_contact(self, params):
        for env_index in range(self.number_env):
            property = self.gym.get_actor_rigid_shape_properties(self._envs.envs[env_index], 0)
            property[6].friction = params[env_index,0]
            property[8].friction = params[env_index,0]
            property[10].friction = params[env_index,0]
            property[12].friction = params[env_index,0]

            property[6].restitution = params[env_index, 1]
            property[8].restitution = params[env_index, 1]
            property[10].restitution = params[env_index, 1]
            property[12].restitution = params[env_index, 1]
        
            self.gym.set_actor_rigid_shape_properties(self._envs.envs[env_index], 0, property)


    def set_DOF(self, params):
        for env_index in range(self.number_env):
            # hip:   0 2 4 6
            # ankle: 1 3 5 7
            property = self.gym.get_actor_dof_properties(self._envs.envs[env_index], 0)

            for i in [0,2,4,6]:
                property[i]['friction'] = params[env_index,0]
                property[i]['damping']  = params[env_index,1]
                property[i]['armature'] = params[env_index,2]

            for i in [1,3,5,7]:
                property[i]['friction'] = params[env_index,3]
                property[i]['damping']  = params[env_index,4]
                property[i]['armature'] = params[env_index,5]

            self.gym.set_actor_dof_properties(self._envs.envs[env_index], 0, property)


    def set_mass(self, params):
        for env_index in range(self.number_env):
            property = self.gym.get_actor_rigid_body_properties(self._envs.envs[env_index], 0)
            for i in [1,3,5,7]:
                property[i].mass = params[env_index,0]
            for i in [2,4,6,8]:
                property[i].mass = params[env_index,1]
            for i in [0]:
                property[i].mass = params[env_index,2]
            self.gym.set_actor_rigid_body_properties(self._envs.envs[env_index], 0, property)



    def step(self, action):
        return self._envs.step(action)
    

    def reset(self):
        # 这里reset需要重置
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





    def step(self, action):
        obs, reward, done, info = self._envs.step(action)
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

    def sample_params_Gaussian(self, params_mean, params_var):
        cov_mat = np.diag(params_var)
        parma_instance = np.empty((self.number_env, len(params_mean)))
        for i in range(self.number_env):
            parma_instance[i] = np.random.multivariate_normal(params_mean,cov_mat)
        self.set_params(parma_instance)
    
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
            tray_shape[0].friction = params[env_index, 2]
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
                    done[i] = True
                    self._envs.reset_idx([i])
                    next_state['obs'][i] = torch.zeros_like(next_state['obs'][i])
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
    # Test environments, Select env_name in ['Ant', 'Ballbalance', 'Cartpole']
    env_name = 'Ant'
    from EvolutionaryAdversarial.algo import SACExpert
    num_envs = 100

    if env_name == 'Ant':
        envs = paramAnt(num_envs, 'cpu',0,headless=False) 
        actor_path='example/example_policy/Ant_DR/actor.pth'
        params=[1.5, 0.3,  0.2, 0.3, 0.1,   0.1, 0.2, 0.1,  0.1, 0.2, 0.25]    
    if env_name == 'Cartpole':
        envs = paramCartpoleFull(num_envs, 'cpu',0,headless=False)
        actor_path='example/example_policy/Cartpole_DR/actor.pth'
        params=[0.3, 0.1, 0.3, 3e-04, 2e-03 ,5e-03, 1e-02, 20, 0.3, 5, 0.6]
    if env_name == 'Ballbalance':
        envs = paramBallBalance(num_envs, 'cpu',0,headless=False)
        actor_path='example/example_policy/Ballablance_DR/actor.pth'
        params=[3, 5 ,1 ,0.3 ,100 ,10 ,5]
    
    actor_policy = SACExpert(
        state_shape = (*envs.observation_space.shape, 1),
        action_shape = (*envs.action_space.shape, 1),
        device=torch.device('cpu'),
        path=actor_path)
    envs.set_params(params)

    for __ in range(1000000):
        obs = envs.reset()
        for _ in range(envs._envs.max_episode_length):
            actions = actor_policy.controller_action(obs['obs'])
            obs, reward, done, info = envs.step(actions)
