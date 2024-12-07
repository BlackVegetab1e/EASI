from IsaacGymEnvs import isaacgymenvs
# import isaacgymenvs
import numpy as np
from isaacgym import gymapi
import torch
from torch.utils.tensorboard import SummaryWriter


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




if __name__ == "__main__":
    num_envs = 100
    envs = paramAnt(num_envs, 'cpu',0,headless=False)
    print(envs.action_space)
    print(envs.observation_space)
    from EvolutionaryAdversarial.algo import SACExpert
    actor_policy = SACExpert(
        state_shape=(60,1),
        action_shape=(8,1),
        device=torch.device('cpu'),
        path='example/example_policy/Ant_DR/actor.pth')

    params= [1.5, 0.3,   0.2, 0.3, 0.1,   0.1, 0.2, 0.1,  0.1, 0.2, 0.25]    

    envs.set_params(params)
    flag = False
    last_angle = 0
    for __ in range(1000000):
        obs = envs.reset()
        reward_sum = 0
        mx_speed = 0
        for _ in range(200):
            
            actions = actor_policy.controller_action(obs['obs'])
            obs, reward, done, info = envs.step(actions)
            reward_sum += reward

        

        

