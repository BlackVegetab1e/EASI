from tqdm import tqdm
import numpy as np
import torch

from .buffer import Buffer



def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.mul_(1.0 - tau)
        t.data.add_(tau * s.data)


def disable_gradient(network):
    for param in network.parameters():
        param.requires_grad = False


def add_random_noise(action, std):
    action += torch.randn(action.shape, device=torch.device("cuda")) * std
    return action


def collect_demo(env, number_of_envs, traj_length , algo, buffer_size, device, std, p_rand, seed=0):

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    buffer = Buffer(
        buffer_size=buffer_size,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=device
    )

    total_return = 0.0
    num_episodes = 0

    state = env.reset()['obs']
    t = 0
    episode_return = 0.0
    isaac_step = 0 


    sum_of_reward = torch.zeros((number_of_envs,1))
    collect_of_reward = torch.zeros((number_of_envs,1))
    collect_of_reward_index = 0
    masks = torch.zeros((number_of_envs,1))
    step_counter = torch.zeros((number_of_envs,1))
    env.max_episode_length = traj_length

    while isaac_step < buffer_size:
    
        t += 1
        isaac_step += number_of_envs

        # print(state)
        action = algo.exploit(state)
        # print(action)
        action = add_random_noise(action, std)

        
        next_state, reward, done, _ = env.step(action)




        next_state = next_state['obs']
        done_cpu = done.cpu()
        reward_cpu = reward.cpu()
        for i in range(number_of_envs):
            masks[i] = 0 if step_counter[i] == env.max_episode_length else done_cpu[i]
            if step_counter[i] == env.max_episode_length or done_cpu[i]:
                total_return += sum_of_reward[i].item()
                num_episodes += 1
                sum_of_reward[i] = 0
                step_counter[i] = 0
            else:
                sum_of_reward[i] += reward_cpu[i]
                step_counter[i] += 1
        mask = torch.tensor(masks, device=device)

        
        buffer.append(state, action, reward, mask, next_state)


        state = next_state
        if isaac_step %int(buffer_size/10) == 0:
            print("now_collect", isaac_step)

    print(f'Mean return of the expert is {total_return / num_episodes}')
    return buffer
