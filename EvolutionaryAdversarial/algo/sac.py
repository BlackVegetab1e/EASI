import os
import torch
from torch import nn
from torch.optim import Adam
import numpy as np
from .base import Algorithm
from EvolutionaryAdversarial.buffer import Buffer
from EvolutionaryAdversarial.utils import soft_update, disable_gradient
from EvolutionaryAdversarial.network import (
    StateDependentPolicy, TwinnedStateActionFunction
)


class SAC(Algorithm):

    def __init__(self, state_shape, action_shape, number_of_envs, device, seed, gamma=0.99,
                 batch_size=256, buffer_size=10**6, lr_actor=3e-4,
                 lr_critic=3e-4, lr_alpha=3e-4, units_actor=(256, 256),
                 units_critic=(256, 256), start_steps=10000, tau=5e-3, FT_Mode = False):
        super().__init__(state_shape, action_shape, device, seed, gamma)

        # Replay buffer.
        self.buffer = Buffer(
            buffer_size=buffer_size,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device
        )

        # Actor.
        self.actor = StateDependentPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_actor,
            hidden_activation=nn.ReLU(inplace=True)
        ).to(device)

        # Critic.
        self.critic = TwinnedStateActionFunction(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_critic,
            hidden_activation=nn.ReLU(inplace=True)
        ).to(device)
        self.critic_target = TwinnedStateActionFunction(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_critic,
            hidden_activation=nn.ReLU(inplace=True)
        ).to(device).eval()

        soft_update(self.critic_target, self.critic, 1.0)
        disable_gradient(self.critic_target)

        # Entropy coefficient.
        self.alpha = 1.0
        # We optimize log(alpha) because alpha should be always bigger than 0.
        self.log_alpha = torch.zeros(1, device=device, requires_grad=True)

        self.FT_mode =  FT_Mode
        if self.FT_mode:
            self.alpha = 0

        # Target entropy is -|A|.
        self.target_entropy = -float(action_shape[0])

        self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_critic)
        self.optim_alpha = torch.optim.Adam([self.log_alpha], lr=lr_alpha)

        self.batch_size = batch_size
        self.start_steps = start_steps
        self.tau = tau
        self.number_of_envs = number_of_envs
        self.sum_of_reward = torch.zeros((number_of_envs,1))
        self.collect_of_reward = torch.zeros((number_of_envs,1))
        self.collect_of_reward_index = 0
        self.masks = torch.zeros((number_of_envs,1))
        self.step_counter = torch.zeros((number_of_envs,1))





    def is_update(self, steps):
        return steps >= max(self.start_steps, self.batch_size)

    def step(self, env, number_of_env, state,  step, data_budget = False, GAT_policy = None):
        """
        data_budget fine tune时使用,只能采集data_budget个数据进行训练,如果现在步数超过data_budget,新的数据将不会进入buffer。
        """
        self.step_counter += 1

        if step <= self.start_steps:

            assert self.device == torch.device(self.device)
            action = 2.0 * torch.rand((number_of_env,) + env.action_space.shape, device = self.device) - 1.0

        else:
            action, log_pi = self.explore(state)
       
        # print(action)
        if GAT_policy is not None:
            GAT_action = GAT_policy.adjust(torch.cat((state, action), dim=1))
            actual_action = GAT_action

            next_state, reward, done, _ = env.step(actual_action)
            next_state = next_state['obs']
        else:
            next_state, reward, done, _ = env.step(action)
            next_state = next_state['obs']
        

        done_cpu = done.cpu()
        reward_cpu = reward.cpu()

        for i in range(self.number_of_envs):
            self.masks[i] = 0 if self.step_counter[i] == env.max_episode_length else done_cpu[i]
            if self.step_counter[i] == env.max_episode_length or done_cpu[i]:
                self.collect_of_reward[self.collect_of_reward_index] = self.sum_of_reward[i]
                self.collect_of_reward_index = (self.collect_of_reward_index + 1) % self.number_of_envs
                self.sum_of_reward[i] = 0
                self.step_counter[i] = 0

            else:
                self.sum_of_reward[i] += reward_cpu[i]
        
        mask = torch.tensor(self.masks, device=self.device)

        if not data_budget:
            self.buffer.append(state, action, reward, mask, next_state)

        # if done:
        #     t = 0
        #     next_state = env.reset()['obs']

        return next_state, self.collect_of_reward, done_cpu

    def update(self, writer):
        self.learning_steps += 1
        states, actions, rewards, dones, next_states = \
            self.buffer.sample(self.batch_size)


        self.update_critic(
            states, actions, rewards, dones, next_states, writer)
        self.update_actor(states, writer)
        self.update_target()

    def update_critic(self, states, actions, rewards, dones, next_states,
                      writer):
        curr_qs1, curr_qs2 = self.critic(states, actions)
        with torch.no_grad():
            next_actions, log_pis = self.actor.sample(next_states)
            next_qs1, next_qs2 = self.critic_target(next_states, next_actions)
            next_qs = torch.min(next_qs1, next_qs2) - self.alpha * log_pis
        target_qs = rewards + (1.0 - dones) * self.gamma * next_qs

        loss_critic1 = (curr_qs1 - target_qs).pow_(2).mean()
        loss_critic2 = (curr_qs2 - target_qs).pow_(2).mean()

        self.optim_critic.zero_grad()
        (loss_critic1 + loss_critic2).backward(retain_graph=False)
        self.optim_critic.step()

        if self.learning_steps % 1000 == 0:
            writer.add_scalar(
                'loss/critic1', loss_critic1.item(), self.learning_steps)
            writer.add_scalar(
                'loss/critic2', loss_critic2.item(), self.learning_steps)

    def update_actor(self, states, writer):
        actions, log_pis = self.actor.sample(states)
        qs1, qs2 = self.critic(states, actions)
        loss_actor = self.alpha * log_pis.mean() - torch.min(qs1, qs2).mean()

        self.optim_actor.zero_grad()
        loss_actor.backward(retain_graph=False)
        self.optim_actor.step()

        entropy = -log_pis.detach_().mean()
        loss_alpha = -self.log_alpha * (self.target_entropy - entropy)

        self.optim_alpha.zero_grad()
        loss_alpha.backward(retain_graph=False)
        self.optim_alpha.step()

        if not self.FT_mode:
            with torch.no_grad():
                self.alpha = self.log_alpha.exp().item()
        # print(self.alpha)
        if self.learning_steps % 1000 == 0:
            writer.add_scalar(
                'loss/actor', loss_actor.item(), self.learning_steps)
            writer.add_scalar(
                'loss/alpha', loss_alpha.item(), self.learning_steps)
            writer.add_scalar(
                'stats/alpha', self.alpha, self.learning_steps)
            writer.add_scalar(
                'stats/entropy', entropy.item(), self.learning_steps)

    def update_target(self):
        soft_update(self.critic_target, self.critic, self.tau)

    def save_models(self, save_dir):
        super().save_models(save_dir)
        # We only save actor to reduce workloads.
        torch.save(
            self.actor.state_dict(),
            os.path.join(save_dir, 'actor.pth')
        )

    def save_all_models(self, save_dir):
        super().save_models(save_dir)
        # We only save actor to reduce workloads.
        torch.save(
            self.actor.state_dict(),
            os.path.join(save_dir, 'actor.pth')
        )
        torch.save(
            self.critic.state_dict(),
            os.path.join(save_dir, 'critic.pth')
        )
        torch.save(
            self.critic_target.state_dict(),
            os.path.join(save_dir, 'critic_target.pth')
        )
    
    def fine_tune_load(self, model_dir):
        self.actor.load_state_dict(torch.load(model_dir+'actor.pth'))
        self.critic.load_state_dict(torch.load(model_dir+'critic.pth'))
        self.critic_target.load_state_dict(torch.load(model_dir+'critic_target.pth'))


class SACExpert(SAC):

    def __init__(self, state_shape, action_shape, device, path,
                 units_actor=(256, 256)):
        self.actor = StateDependentPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_actor,
            hidden_activation=nn.ReLU(inplace=True)
        ).to(device)
        self.actor.load_state_dict(torch.load(path))

        disable_gradient(self.actor)
        self.device = device
    

    def step(self, env, number_of_env, state, GAT_policy = None):
        action = self.exploit(state)


        if GAT_policy is not None:
            GAT_action = GAT_policy.adjust(torch.cat((state, action), dim=1))
            actual_action = GAT_action
            next_state, reward, done, _ = env.step(actual_action)
            next_state = next_state['obs']
            # print('adjusted')
        else:
            next_state, reward, done, _ = env.step(action)
            next_state = next_state['obs']

        done_cpu = done.cpu()
        reward_cpu = reward.cpu()
        

        return next_state, reward_cpu, done_cpu
    def controller_action(self, state, max = 1):
        action = max*self.exploit(state)
        return action