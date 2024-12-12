import os
import numpy as np
import torch


class SerializedBuffer:

    def __init__(self, path, device):
        tmp = torch.load(path)
        self.buffer_size = self._n = tmp['state'].size(0)
        self.device = device

        self.states = tmp['state'].clone().to(self.device)
        self.actions = tmp['action'].clone().to(self.device)
        self.rewards = tmp['reward'].clone().to(self.device)
        self.dones = tmp['done'].clone().to(self.device)
        self.next_states = tmp['next_state'].clone().to(self.device)

    def sample(self, batch_size):
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)

        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.next_states[idxes]
        )


class Buffer(SerializedBuffer):

    def __init__(self, buffer_size, state_shape, action_shape, device):
        self._n = 0
        self._p = 0
        self.buffer_size = buffer_size
        self.device = device

        self.states = torch.empty(
            (buffer_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty(
            (buffer_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (buffer_size, *state_shape), dtype=torch.float, device=device)

    def append(self, state, action, reward, done, next_state):
        for i in range(0, state.shape[0]):
            self.states[self._p].copy_(state[i])
            self.actions[self._p].copy_(action[i])
            self.rewards[self._p] = float(reward[i])
            self.dones[self._p] = float(done[i])
            self.next_states[self._p].copy_(next_state[i])

            self._p = (self._p + 1) % self.buffer_size
            self._n = min(self._n + 1, self.buffer_size)

    

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        torch.save({
            'state': self.states.clone().cpu(),
            'action': self.actions.clone().cpu(),
            'reward': self.rewards.clone().cpu(),
            'done': self.dones.clone().cpu(),
            'next_state': self.next_states.clone().cpu(),
        }, path)

    def delete_buffer(self):
        self._n = 0
        self._p = 0
 
        self.states = torch.empty_like(self.states)
        self.actions = torch.empty_like(self.actions)
        self.rewards = torch.empty_like(self.rewards)
        self.dones = torch.empty_like(self.dones)
        self.next_states = torch.empty_like(self.next_states)
        print('buffer_deleted')


class RolloutBuffer:

    def __init__(self, buffer_size, state_shape, action_shape, device, mix=1):
        self._n = 0
        self._p = 0
        self.mix = mix
        self.buffer_size = buffer_size
        self.total_size = mix * buffer_size

        self.states = torch.empty(
            (self.total_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty(
            (self.total_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.log_pis = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (self.total_size, *state_shape), dtype=torch.float, device=device)

    def append(self, state, action, reward, done, log_pi, next_state):
        
        state_s = torch.unsqueeze(state,dim=-1)
        action_s = torch.unsqueeze(action,dim=-1)
        next_state_s = torch.unsqueeze(next_state,dim=-1)

        self.states[self._p].copy_(state_s)
        self.actions[self._p].copy_(action_s)
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.log_pis[self._p] = float(log_pi)
        self.next_states[self._p].copy_(next_state_s)

        self._p = (self._p + 1) % self.total_size
        self._n = min(self._n + 1, self.total_size)

        # print(self._p)
        # print(self.buffer_size)

    def get(self):
        assert self._p % self.buffer_size == 0
        start = (self._p - self.buffer_size) % self.total_size
        idxes = slice(start, start + self.buffer_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes]
        )

    def sample(self, batch_size):

        assert self._p % self.buffer_size == 0
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes]
        )



class StateTransBuffer:

    def __init__(self, buffer_size, state_shape, action_shape, device):
        self._n = 0
        self._p = 0

        self.buffer_size = buffer_size
        self.total_size = buffer_size

        self.states = torch.empty(
            (self.total_size, *state_shape), dtype=torch.float, device=device)

        self.actions = torch.empty(
            (self.total_size, *action_shape), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (self.total_size, *state_shape), dtype=torch.float, device=device)

    def append(self, state, action, next_state):
        
        state = state.reshape(-1,1)
        action = action.reshape(-1,1)
        next_state = next_state.reshape(-1,1)

        
        self.states[self._p].copy_(state)
        self.actions[self._p].copy_(action)   
        self.next_states[self._p].copy_(next_state)

        self._p = (self._p + 1) 
        self._n = min(self._n + 1, self.total_size)

    def get(self):
        # print(self._p, self.buffer_size)
        assert self._p == self.buffer_size 
        # 因为这个玩意，当做NN的输入，长度必须符合规范
        # 所以这里的长度是有要求的。这其实是个很大的问题
        # 未来能否使用RNN，或者LSTM，将这个玩意变成轨迹可变长度的?

        return self.states, self.actions, self.next_states

    def to_tensor(self):
        all_in_one = torch.cat([self.states.flatten(),self.actions.flatten(),self.next_states.flatten()])
    


        # np.savetxt('./obj/all_in_one.csv',all_in_one.squeeze().cpu().detach().numpy(),fmt='%.2f',delimiter=',')
        # np.savetxt('./obj/states.csv',self.states.squeeze().cpu().detach().numpy(),fmt='%.2f',delimiter=',')
        # np.savetxt('./obj/actions.csv',self.actions.squeeze().cpu().detach().numpy(),fmt='%.2f',delimiter=',')
        # np.savetxt('./obj/next_states.csv',self.next_states.squeeze().cpu().detach().numpy(),fmt='%.2f',delimiter=',')
        # print(all_in_one.shape)
        # assert(False)
        return all_in_one
    
    def sample(self, batch_size):
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.next_states[idxes]
        )

        
class RefBufferOfTragectory:

    def __init__(self, buffer_size, device):
        
        self.buffer_size = buffer_size
        self.now_size = 0
        self.now_at = 0

        self.ref_buffer = []
        self.device = device
    

# append(tragectory, action_env, done, log_pi_env, next_tragectory)

    def append(self, tragectory):
        if(self.now_size < self.buffer_size):
            self.ref_buffer.append(tragectory)
            self.now_size += 1
            
    
        else:
            self.now_at = self.now_at  % self.buffer_size
            self.ref_buffer[self.now_at] = tragectory 
        # print(action_env[0].shape)
        # print(self.action_env[self.now_at].shape)

        self.now_at+=1

    # def get(self):

  
    #     assert self.now_at % self.buffer_size == 0
    #     start = (self.now_at - self.buffer_size) % self.buffer_size
    #     idxes = slice(start, start + self.buffer_size)
    #     return (
    #         self.ref_buffer[idxes],
    #         self.action_env[idxes],
    #         self.ref_buffer_next[idxes],
    #         self.old_params[idxes],
    #         self.log_pi_env[idxes]
    #     )
    
    def sample(self, batch_size):
        batch_size_per_trag = int(batch_size/self.now_size)
        single_x, single_a, single_x_n = self.ref_buffer[0].sample(batch_size_per_trag)
        x = single_x.clone().detach()
        a = single_a.clone().detach()
        x_n = single_x_n.clone().detach()
        for i in range(1, self.now_size):
            single_x, single_a, single_x_n = self.ref_buffer[i].sample(batch_size_per_trag)
            single_x = single_x.clone().detach()
            single_a = single_a.clone().detach()
            single_x_n = single_x_n.clone().detach()
           
            x = torch.cat((x,single_x), dim=0)
            
            a = torch.cat((a,single_a), dim=0)
            x_n = torch.cat((x_n,single_x_n), dim=0)
        

        return x, a, x_n

    def save(self, path, traj_length):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        
        all_steps = self.now_size * self.ref_buffer[0].buffer_size
       
        state_length = self.ref_buffer[0].states.shape[1]
        action_length = self.ref_buffer[0].actions.shape[1]
        state_in_one = torch.empty((all_steps,state_length),device=torch.device(self.device))
        next_state_in_one = torch.empty((all_steps,state_length),device=torch.device(self.device))
        action_in_one = torch.empty((all_steps,action_length),device=torch.device(self.device))

        begin = 0
        end = 0
        for i in range(self.now_size):
            end += self.ref_buffer[i].buffer_size

            state_in_one[begin:end].copy_(self.ref_buffer[i].states.squeeze())
            next_state_in_one[begin:end].copy_(self.ref_buffer[i].next_states.squeeze())

            action_in_one[begin:end].copy_(self.ref_buffer[i].actions.squeeze(-1))
            begin += self.ref_buffer[i].buffer_size


        check = True
        for i in range(self.now_size):
            rand = torch.randint(0,traj_length, (1,))
            index = i*traj_length + rand.item()
            
            b1 = torch.equal(state_in_one[index], self.ref_buffer[i].states[rand].squeeze())
            b2 = torch.equal(next_state_in_one[index],self.ref_buffer[i].next_states[rand].squeeze())
            b3 = torch.equal(action_in_one[index].squeeze(),self.ref_buffer[i].actions[rand].squeeze())
            if not(b1 and b2 and b3):
                check = False
        
        if check:
            print('Check Done')
            print(path)
            torch.save({
            'states': state_in_one.clone().cpu(),
            'actions': action_in_one.clone().cpu(),
            'next_states': next_state_in_one.clone().cpu(),
        }, path)
        else:
            print('something error')

        
class ExpertBuffer():
    def __init__(self, expert_path, device):
    
        tmp = torch.load(expert_path)
  
        self.states = tmp['states'].to(torch.device(device))
        self.actions = tmp['actions'].to(torch.device(device))
        self.next_states = tmp['next_states'].to(torch.device(device))
        self.BufferSize = self.states.shape[0]
    
    def sample(self, batch_size):
        idxes = np.random.randint(low=0, high=self.BufferSize, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.next_states[idxes]
        )

        




        
