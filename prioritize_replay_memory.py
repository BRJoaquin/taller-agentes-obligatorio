import random
from collections import namedtuple
import random
import torch
import numpy as np


class PrioritizedReplayMemory:
    # Prioritized replay memory
    # buffer_size: size of the memory buffer
    # device: device to send the tensors to
    # alpha: how much prioritization is used (0 - no prioritization, 1 - full prioritization)
    # beta: To what degree importance weights are used (0 - no corrections, 1 - full correction)
    def __init__(self, buffer_size, device, alpha=0.6, beta=0.4):
        self.device = device
        self.buffer_size = buffer_size
        self.memory = []
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.alpha = alpha
        self.beta = beta
        self.pos = 0

    def add(self, state, action, reward, done, next_state, x_pos, time, next_x_pos, next_time):
        max_prio = self.priorities.max() if self.memory else 1.0

        # Create a transition and send it to the device
        transition = {
            "state": torch.tensor(state).to(self.device),
            "action": torch.tensor(action).to(self.device),
            "reward": torch.tensor(reward).to(self.device),
            "done": torch.tensor(done).to(self.device),
            "next_state": torch.tensor(next_state).to(self.device),
            "x_pos": torch.tensor(x_pos).to(self.device),
            "time": torch.tensor(time).to(self.device),
            "next_x_pos": torch.tensor(next_x_pos).to(self.device),
            "next_time": torch.tensor(next_time).to(self.device),
        }

        # In case the buffer is full, remove the oldest transition
        if len(self.memory) >= self.buffer_size:
            self.memory.pop(0)

        # Add the transition to the memory
        self.memory.append(transition)

        # Update the priority of the current position with the max priority
        self.priorities[self.pos] = max_prio
        # Update the position
        self.pos = (self.pos + 1) % self.buffer_size

    def sample(self, batch_size):
        # Determine the priorities to consider based on the size of the memory buffer
        if len(self.memory) == self.buffer_size:
            prios = self.priorities
        else:
            prios = self.priorities[: self.pos]

        # Calculate the probabilities of each transition to be chosen
        probs = prios**self.alpha
        probs /= probs.sum()

        # Choose a set of indices according to the calculated probabilities
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        # Use these indices to select a batch of transitions
        samples = [self.memory[idx] for idx in indices]

        # Calculate the weights for the importance sampling correction
        total = len(self.memory)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        states = torch.stack([transition["state"] for transition in samples])
        actions = torch.stack([transition["action"] for transition in samples])
        rewards = torch.stack([transition["reward"] for transition in samples])
        dones = torch.stack([transition["done"] for transition in samples])
        next_states = torch.stack([transition["next_state"] for transition in samples])
        x_pos = torch.stack([transition["x_pos"] for transition in samples])
        time = torch.stack([transition["time"] for transition in samples])
        next_x_pos = torch.stack([transition["next_x_pos"] for transition in samples])
        next_time = torch.stack([transition["next_time"] for transition in samples])

        return states, actions, rewards, dones, next_states, x_pos, time, next_x_pos, next_time, indices, weights

    # Update the priorities of a batch of transitions
    def update_priorities(self, batch_indices, batch_priorities):
        # Iterate over the provided indices and their corresponding new priorities
        for idx, prio in zip(batch_indices, batch_priorities):
            # Update the priority of the transition at the specified index
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.memory)
