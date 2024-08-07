import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD, RMSprop

from .ppo import PPO
from gail_airl_ppo.network import GAILDiscrim


class GAIL(PPO):

    def __init__(self, buffer_exp, state_shape,tragectorch_length, action_shape,
                 device, seed,gamma=0.995, rollout_length=50000, mix_buffer=1,
                 batch_size=64, lr_actor=3e-4, lr_critic=3e-4, lr_disc=3e-4,
                 units_actor=(64, 64), units_critic=(64, 64),
                 units_disc=(100, 100), epoch_ppo=50, epoch_disc=10,
                 clip_eps=0.2, lambd=0.97, coef_ent=0.0):
        super().__init__(
            state_shape = state_shape, 
            action_shape = action_shape, 
            trajectory_length= tragectorch_length,
            device = device, 
            seed = seed, 
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
        self.disc = GAILDiscrim(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_disc,
            hidden_activation=nn.ReLU()
        ).to(device)

        self.learning_steps_disc = 0
        self.optim_disc = RMSprop(self.disc.parameters(), lr=lr_disc)
        self.batch_size = batch_size
        self.epoch_disc = epoch_disc

    def update(self, writer, ref_tragectory_buffer):
        self.learning_steps += 1

        for _ in range(self.epoch_disc):
            self.learning_steps_disc += 1

            # Samples from current policy's trajectories.
            states, actions, next_states = ref_tragectory_buffer.sample_s_a(self.batch_size)
            # sampled = self.buffer_exp.sample(self.batch_size)

            # states, actions = sampled[:2]
            # next_states = sampled[4]

            # Samples from expert's demonstrations.
            states_exp, actions_exp, next_states_exp = self.buffer_exp.sample(self.batch_size)
            
        
            # states_exp, actions_exp, next_states_exp = ref_tragectory_buffer.sample_s_a(self.batch_size)
            
            # Update discriminator.
            # print("Update discriminator.")
            self.update_disc(states, actions, next_states, \
                             states_exp, actions_exp, next_states_exp, writer)


    def update_disc(self, states, actions, next_states, states_exp, actions_exp, next_states_exp, writer):
        # Output of discriminator is (-inf, inf), not [0, 1].
        states = states.squeeze(-1)
        actions = actions.squeeze(-1)
        next_states = next_states.squeeze(-1)

        self.optim_disc.zero_grad()

        WEIGHT_CLIP = 0.01
        for p in self.disc.parameters():
            p.data.clamp_(-WEIGHT_CLIP,WEIGHT_CLIP)

        # one = torch.tensor([1.],device=self.device)
        # mone = one * -1

        # Train discriminator
        # WGAN - Training discriminator more iterations than generator
        # Train with real images
        expert_d    = self.disc(states_exp, actions_exp, next_states_exp)
        expert_loss = - expert_d.mean()
        # d_loss_real.backward(one)

        # Train with fake images

        policy_d    = self.disc(states, actions, next_states)
        policy_loss = policy_d.mean()

        # d_loss_fake.backward(mone)

        d_loss = 0.5*(expert_loss + policy_loss)
        d_loss.backward()
        self.optim_disc.step()

        writer.add_scalar(
            'loss/loss_fake', expert_loss.item(), self.learning_steps_disc)
        writer.add_scalar(
            'loss/loss_real', policy_loss.item(), self.learning_steps_disc)
        writer.add_scalar(
            'loss/loss',      d_loss.item(), self.learning_steps_disc)



