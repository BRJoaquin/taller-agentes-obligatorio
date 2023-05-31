import random
from collections import namedtuple
import random
import torch


class ReplayMemory:
    def __init__(self, buffer_size):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.buffer_size = buffer_size
        self.memory = []

    def add(self, state, action, reward, done, next_state):
        transition = {
            "state": torch.tensor(state).to(self.device),
            "action": torch.tensor(action).to(self.device),
            "reward": torch.tensor(reward).to(self.device),
            "done": torch.tensor(done).to(self.device),
            "next_state": torch.tensor(next_state).to(self.device),
        }

        if len(self.memory) >= self.buffer_size:
            self.memory.pop(0)

        self.memory.append(transition)


    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)

        states = torch.stack([transition["state"] for transition in batch])
        actions = torch.stack([transition["action"] for transition in batch])
        rewards = torch.stack([transition["reward"] for transition in batch])
        dones = torch.stack([transition["done"] for transition in batch])
        next_states = torch.stack([transition["next_state"] for transition in batch])

        return states, actions, rewards, dones, next_states

    def __len__(self):
        return len(self.memory)
