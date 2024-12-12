import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD, RMSprop


from EvolutionaryAdversarial.network import GAILDiscrim
from EvolutionaryAdversarial.buffer import RolloutBuffer, StateTransBuffer, RefBufferOfTragectory
from EvolutionaryAdversarial.network import StateIndependentPolicy, StateFunction, ParamGenerateNet





class StateTransDiscriminator():

    def __init__(self, buffer_exp, state_shape,  action_shape,
                 device, batch_size=64, lr_disc=3e-4, units_disc=(100, 100), epoch_disc=10):


        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device

        # Expert's buffer.
        self.buffer_exp = buffer_exp

        # Discriminator.
        self.disc = GAILDiscrim(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_disc,
            hidden_activation=nn.ReLU()
        ).to(device)

        self.learning_steps = 0
        self.learning_steps_disc = 0
        self.optim_disc = RMSprop(self.disc.parameters(), lr=lr_disc)
        self.batch_size = batch_size
        self.epoch_disc = epoch_disc



    def update(self, writer, ref_tragectory_buffer):
        self.learning_steps += 1

        for _ in range(self.epoch_disc):
            self.learning_steps_disc += 1

            # Samples from current policy's trajectories.
            states, actions, next_states = ref_tragectory_buffer.sample(self.batch_size)
            # sampled = self.buffer_exp.sample(self.batch_size)

            # states, actions = sampled[:2]
            # next_states = sampled[4]

            # Samples from expert's demonstrations.
            states_exp, actions_exp, next_states_exp = self.buffer_exp.sample(self.batch_size)
            
            # debug tips: 
            # states_exp, actions_exp, next_states_exp = ref_tragectory_buffer.sample_s_a(self.batch_size)
            
            # Update discriminator.
            # print("Update discriminator.")
            # print(states)
            # print(states_exp)
            self.update_disc(states, actions, next_states, \
                             states_exp, actions_exp, next_states_exp, writer)


    def update_disc(self, states, actions, next_states, states_exp, actions_exp, next_states_exp, writer):
        states = states.squeeze(-1)
        actions = actions.squeeze(-1)
        next_states = next_states.squeeze(-1)

        self.optim_disc.zero_grad()

        WEIGHT_CLIP = 0.01
        for p in self.disc.parameters():
            p.data.clamp_(-WEIGHT_CLIP,WEIGHT_CLIP)

        # Train discriminator
        # WGAN - Training discriminator more iterations than generator
        # Train with real
        expert_d    = self.disc(states_exp, actions_exp, next_states_exp)
        expert_loss = - expert_d.mean()

        # Train with fake
        policy_d   = self.disc(states, actions, next_states)
        policy_loss = policy_d.mean()

        d_loss = 0.5*(expert_loss + policy_loss)
        d_loss.backward()
        self.optim_disc.step()

        writer.add_scalar(
            'loss/loss_fake', expert_loss.item(), self.learning_steps_disc)
        writer.add_scalar(
            'loss/loss_real', policy_loss.item(), self.learning_steps_disc)
        writer.add_scalar(
            'loss/loss',      d_loss.item(), self.learning_steps_disc)

    def is_update(self, step):
        return step % self.rollout_length == 0

    def step(self,  envs,  expert_policy ,step_length, params, action_rate = 1):
        # 使用一组参数，跑出一组轨迹
        # trajectory是一堆状态转移，step里面使用状态转移求出参数变化量
        # 使用变化后的参数去跑expert策略
        # 返回值是调整过的环境中，跑出来的轨迹
        env_number = envs.number_env
        envs.reset_isaacgym_env(range(env_number))
        
        # print(trajectory_states.shape)

        if params is not None:
            envs.set_params(params=params)

        next_tragectory = [StateTransBuffer(step_length, self.state_shape, \
                                       self.action_shape, self.device) for _ in range(env_number)]

        state = envs.reset()['obs']

        for step in range(step_length):
            if expert_policy is not None:
                action, log_pi = expert_policy.explore(state)
            else:
                action = 0.0*torch.ones((env_number,envs.action_space.shape[0]))
                
            action *= action_rate
            next_state, reward, done, _ = envs.step(action)
            next_state = next_state['obs']
            # print(next_state[0:10,1])
            
            for i in range(env_number):
                next_tragectory[i].append(state[i], action[i], next_state[i])
            state = next_state
        
        return next_tragectory

