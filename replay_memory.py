import random
from collections import namedtuple
import random
import torch


class ReplayMemory:
    def __init__(self, buffer_size, device):
        self.device = device
        self.buffer_size = buffer_size
        self.memory = []

    # Add a transition to the memory
    def add(self, state, action, reward, done, next_state):
        # Create a transition and send it to the device
        transition = {
            "state": torch.tensor(state).to(self.device),
            "action": torch.tensor(action).to(self.device),
            "reward": torch.tensor(reward).to(self.device),
            "done": torch.tensor(done).to(self.device),
            "next_state": torch.tensor(next_state).to(self.device),
        }

        # In case the buffer is full, remove the oldest transition
        if len(self.memory) >= self.buffer_size:
            self.memory.pop(0)

        # Add the transition to the memory
        self.memory.append(transition)

    # Sample a batch of transitions
    def sample(self, batch_size):
        # Sample a batch of transitions
        batch = random.sample(self.memory, batch_size)

        # Create a batch of tensors
        states = torch.stack([transition["state"] for transition in batch])
        actions = torch.stack([transition["action"] for transition in batch])
        rewards = torch.stack([transition["reward"] for transition in batch])
        dones = torch.stack([transition["done"] for transition in batch])
        next_states = torch.stack([transition["next_state"] for transition in batch])

        return states, actions, rewards, dones, next_states

    # Return the number of transitions in the memory
    def __len__(self):
        return len(self.memory)
