import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from .base import Algorithm
from .sac import SACExpert
from gail_airl_ppo.buffer import RolloutBuffer
from gail_airl_ppo.network import StateIndependentPolicy, StateFunction
from torch.utils.tensorboard import SummaryWriter
from gail_airl_ppo.network.disc import GARATDiscrim

import os
import numpy as np



def calculate_gae(values, rewards, dones, next_values, gamma, lambd):
    # Calculate TD errors.
    deltas = rewards + gamma * next_values * (1 - dones) - values
    # Initialize gae.
    gaes = torch.empty_like(rewards)

    # Calculate gae recursively from behind.
    gaes[-1] = deltas[-1]
    for t in reversed(range(rewards.size(0) - 1)):
        gaes[t] = deltas[t] + gamma * lambd * (1 - dones[t]) * gaes[t + 1]

    return gaes + values, (gaes - gaes.mean()) / (gaes.std() + 1e-8)



# GAT 的STATE即为正常RL算法的S+A，输出动作A，奖励为模仿奖励。
class GARAT_PPO(Algorithm):

    def __init__(self, state_shape, state_action_shape, action_shape, number_of_envs, device, seed, action_policy_dir, gamma=0.995,
                 rollout_length=2048, mix_buffer=20, lr_actor=3e-4,
                 lr_critic=3e-4, units_actor=(64, 64), units_critic=(64, 64),
                 epoch_ppo=10, clip_eps=0.2, lambd=0.97, coef_ent=0.0,
                 max_grad_norm=10.0):
        super().__init__(state_shape, action_shape, device, seed, gamma)
        
        # print(state_shape)
        # Rollout buffer.
        self.buffer = RolloutBuffer(
            buffer_size=rollout_length*number_of_envs,
            state_shape=state_action_shape,
            action_shape=action_shape,
            device=device,
            mix=mix_buffer
        )

        # Actor.
        self.actor = StateIndependentPolicy(
            state_shape=state_action_shape,
            action_shape=action_shape,
            hidden_units=units_actor,
            hidden_activation=nn.Tanh()
        ).to(device)

        # Critic.
        self.critic = StateFunction(
            state_shape=state_action_shape,
            hidden_units=units_critic,
            hidden_activation=nn.Tanh()
        ).to(device)

        if action_policy_dir is not None:
            self.action_policy = SACExpert(
            state_shape=state_shape,
            action_shape=action_shape,
            device=torch.device(device),
            path=action_policy_dir)
        else:
            self.action_policy = None




        self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_critic)

        self.learning_steps_ppo = 0
        self.rollout_length = rollout_length
        self.epoch_ppo = epoch_ppo
        self.clip_eps = clip_eps
        self.lambd = lambd
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm


        self.step_counter = torch.zeros((number_of_envs,1))
        self.number_of_envs = number_of_envs

    def is_update(self, step):
        return step % (self.rollout_length*self.number_of_envs) == 0

    def step(self, env, number_of_env, state, step):
        self.step_counter += 1
        origin_action = self.action_policy.controller_action(state)
 
        GAT_state = torch.cat((state,origin_action),dim=1)
        # print(GAT_state[0])
        GAT_action, log_pi = self.explore(GAT_state)
        actual_action = GAT_action
        # print(actual_action)
        next_state, reward, done, _ = env.step(actual_action)

        mask = torch.zeros((number_of_env,1))

        for i in range(number_of_env):
            mask[i] = False if self.step_counter[i] == env.max_episode_length else done[i]
            if done[i]:
                self.step_counter[i] = 0

        # print(GAT_state, GAT_action, reward, mask, log_pi, next_state)
        
        next_state = next_state['obs']

        GAT_next_state = torch.cat((next_state, self.action_policy.controller_action(next_state)),dim=1)

        for i in range(number_of_env):
            self.buffer.append(GAT_state[i], GAT_action[i], reward[i], mask[i], log_pi[i], GAT_next_state[i])



        return next_state, reward

    def update(self, writer):
        self.learning_steps += 1
        states, actions, rewards, dones, log_pis, next_states = \
            self.buffer.get()
        self.update_ppo(
            states, actions, rewards, dones, log_pis, next_states, writer)

    def update_ppo(self, states, actions, rewards, dones, log_pis, next_states,
                   writer):
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)

        targets, gaes = calculate_gae(
            values, rewards, dones, next_values, self.gamma, self.lambd)

        for _ in range(self.epoch_ppo):
            self.learning_steps_ppo += 1
            self.update_critic(states, targets, writer)
            self.update_actor(states, actions, log_pis, gaes, writer)

    def update_critic(self, states, targets, writer):
        loss_critic = (self.critic(states) - targets).pow_(2).mean()

        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            writer.add_scalar(
                'loss/critic', loss_critic.item(), self.learning_steps)

    def update_actor(self, states, actions, log_pis_old, gaes, writer):
        log_pis = self.actor.evaluate_log_pi(states, actions)
        entropy = -log_pis.mean()

        ratios = (log_pis - log_pis_old).exp_()
        loss_actor1 = -ratios * gaes
        loss_actor2 = -torch.clamp(
            ratios,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * gaes
        loss_actor = torch.max(loss_actor1, loss_actor2).mean()

        self.optim_actor.zero_grad()
        (loss_actor - self.coef_ent * entropy).backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            writer.add_scalar(
                'loss/actor', loss_actor.item(), self.learning_steps)
            writer.add_scalar(
                'stats/entropy', entropy.item(), self.learning_steps)

    def save_models(self, save_dir):
        super().save_models(save_dir)
        # We only save actor to reduce workloads.
        torch.save(
            self.actor.state_dict(),
            os.path.join(save_dir, 'actor.pth')
        )



    def load_actor(self, save_dir):
        self.actor.load_state_dict(torch.load(save_dir))


    def adjust(self, state):
        return self.exploit(state)








class GARAT(GARAT_PPO):
# GAT使用S和A作为输入，输出修正后的A，修正后的A作用于sim环境。
# 希望使修正后的A，形成的sa(源a)s'，和真实世界的sas'更加相似。
    def __init__(self, buffer_exp, state_shape, action_shape, number_of_envs,
                 device, seed, action_policy_dir, gamma=0.99, rollout_length=50000, mix_buffer=1,
                 batch_size=64, lr_actor=3e-4, lr_critic=3e-4, lr_disc=3e-4,
                 units_actor=(64, 64), units_critic=(64, 64),
                 units_disc=(100, 100), epoch_ppo=10, epoch_disc=10,
                 clip_eps=0.2, lambd=0.95, coef_ent=0.0):
        super().__init__(
            state_shape = state_shape, 
            state_action_shape = (state_shape[0] + action_shape[0], 1),
            action_shape = action_shape, 
            number_of_envs=number_of_envs,
            device = device, 
            seed = seed, 
            action_policy_dir = action_policy_dir,
            gamma = gamma, 
            rollout_length = rollout_length,
            mix_buffer = mix_buffer, 
            lr_actor = lr_actor, 
            lr_critic = lr_critic, 
            units_actor = units_actor, 
            units_critic = units_critic,
            epoch_ppo=epoch_ppo, 
            clip_eps = clip_eps, 
            lambd=lambd, 
            coef_ent=coef_ent, 
        )

        # Expert's buffer.
        self.buffer_exp = buffer_exp

        # Discriminator.
        self.disc = GARATDiscrim(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_disc,
            hidden_activation=nn.ReLU()
        ).to(device)

        self.learning_steps_disc = 0
        # WGAIL使用这个：self.optim_disc = RMSprop(self.disc.parameters(), lr=lr_disc)
        self.optim_disc = Adam(self.disc.parameters(), lr=lr_disc)
        self.batch_size = batch_size
        self.epoch_disc = epoch_disc


    def update(self, writer):
        self.learning_steps += 1

        for _ in range(self.epoch_disc):
            self.learning_steps_disc += 1

            # Samples from current policy's trajectories.
            GAT_states, GAT_actions, _, _, _, GAT_next_states = self.buffer.sample(self.batch_size)

            # 因为PPO中保存的state是sa的组合，a是调整过的a，所以这里要还原出sas，从PPOs中剥离出s和a
            states = GAT_states[:, :self.state_shape[0]]
            actions = GAT_states[:, -self.action_shape[0]:]
            next_states = GAT_next_states[:, :self.state_shape[0]]
            # sampled = self.buffer_exp.sample(self.batch_size)


            # Samples from expert's demonstrations.
            states_exp, actions_exp, next_states_exp = self.buffer_exp.sample(self.batch_size)
            
        
            # states_exp, actions_exp, next_states_exp = ref_tragectory_buffer.sample_s_a(self.batch_size)
            
            # Update discriminator.
            # print("Update discriminator.")
            self.update_disc(states, actions, next_states, \
                             states_exp, actions_exp, next_states_exp, writer)

        # We don't use reward signals here,
        GAT_states, GAT_actions, _, dones, log_pis, GAT_next_states = self.buffer.get()

        # 因为PPO中保存的state是sa的组合，a是调整过的a，所以这里要还原出sas，从PPOs中剥离出s和a
        states = GAT_states[:, :self.state_shape[0]]
        actions = GAT_actions[:, -self.action_shape[0]:]
        next_states = GAT_next_states[:, :self.state_shape[0]]

        # Calculate rewards.
        rewards = self.disc.calculate_reward(states, actions, next_states)

        # Update PPO using estimated rewards.
        GAT_states = GAT_states.squeeze(-1)
        GAT_actions = GAT_actions.squeeze(-1)
        GAT_next_states = GAT_next_states.squeeze(-1)
        self.update_ppo(
            GAT_states, GAT_actions, rewards, dones, log_pis, GAT_next_states, writer)


        


    def update_disc(self, states, actions, next_states, states_exp, actions_exp, next_states_exp, writer):
        # Output of discriminator is (-inf, inf), not [0, 1].
        states = states.squeeze(-1)
        actions = actions.squeeze(-1)
        next_states = next_states.squeeze(-1)
        # print('pi_______________________')
        # print(states[0], actions[0], next_states[0])
        # print('exp______________________')
        # print(states_exp[0], actions_exp[0], next_states_exp[0])
        # Output of discriminator is (-inf, inf), not [0, 1].
        logits_pi = self.disc(states, actions, next_states)
        logits_exp = self.disc(states_exp, actions_exp, next_states_exp)

        # Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)].
        loss_pi = -F.logsigmoid(-logits_pi).mean()
        loss_exp = -F.logsigmoid(logits_exp).mean()
        loss_disc = loss_pi + loss_exp

        self.optim_disc.zero_grad()
        loss_disc.backward()
        self.optim_disc.step()

        if self.learning_steps_disc % self.epoch_disc == 0:
            writer.add_scalar(
                'loss/disc', loss_disc.item(), self.learning_steps)

            # Discriminator's accuracies.
            with torch.no_grad():
                acc_pi = (logits_pi < 0).float().mean().item()
                acc_exp = (logits_exp > 0).float().mean().item()
            writer.add_scalar('stats/acc_pi', acc_pi, self.learning_steps)
            writer.add_scalar('stats/acc_exp', acc_exp, self.learning_steps)




class GARATTrainer:

    def __init__(self, env,algo , number_of_env,  log_dir, seed=0, num_steps=10**5,
                 eval_interval=10**3, num_eval_episodes=5):
        super().__init__()

        # Env to collect samples.
        self.env = env
        self.number_of_env = number_of_env
        # self.env.seed(seed)


        self.algo =  algo
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













    