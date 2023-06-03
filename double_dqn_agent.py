import torch
import torch.nn as nn
import torch.nn.functional as F
from replay_memory import ReplayMemory
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
import numpy as np
from abstract_agent import Agent


class DoubleDQNAgent(Agent):
    def __init__(
        self,
        gym_env,
        model_a,
        model_b,
        obs_processing_func,
        memory_buffer_size,
        batch_size,
        learning_rate,
        gamma,
        epsilon_i,
        epsilon_f,
        epsilon_anneal_time,
        epsilon_decay,
        episode_block,
        sync_target=100,
        steps_before_training=10000,
    ):
        super().__init__(
            gym_env,
            obs_processing_func,
            memory_buffer_size,
            batch_size,
            learning_rate,
            gamma,
            epsilon_i,
            epsilon_f,
            epsilon_anneal_time,
            epsilon_decay,
            episode_block,
            steps_before_training,
        )

        # Assign models to agent (and send them to the proper device)
        self.q_a = model_a.to(self.device)
        self.q_b = model_b.to(self.device)

        # Assign a loss function (MSE) (and send it to the proper device)
        self.loss_function = nn.MSELoss().to(self.device)

        # Assign an optimizer for each model (Adam)
        self.optimizer_A = torch.optim.Adam(self.q_a.parameters(), lr=learning_rate)
        self.optimizer_B = torch.optim.Adam(self.q_b.parameters(), lr=learning_rate)

        # Assign a sync target
        self.sync_target = sync_target

    def select_action(self, state, current_steps, train=True):
        state_tensor = self.state_processing_function(state).to(self.device)
        with torch.no_grad():
            q_values = self.q_a(state_tensor) + self.q_b(state_tensor)

        if train:
            epsilon = self.compute_epsilon(current_steps)
            if np.random.random() < epsilon:
                action = self.env.action_space.sample()
            else:
                action = torch.argmax(q_values).item()
        else:
            action = torch.argmax(q_values).item()
        return action

    def update_weights(self, total_steps):
        if len(self.memory) > self.batch_size:
            # Get a minibatch from memory. Resulting in tensors of states, actions, rewards, termination flags, and next states.
            states, actions, rewards, dones, next_states = self.memory.sample(
                self.batch_size
            )

            # Randomly update Q_a or Q_b using the other to calculate the value of the next states.
            if np.random.random() < 0.5:
                q_actual = self.q_a(states).gather(1, actions.unsqueeze(-1))
                max_q_next_state = self.q_b(next_states).detach().max(1)[0]
                self.optimizer = self.optimizer_A
            else:
                q_actual = self.q_b(states).gather(1, actions.unsqueeze(-1))
                max_q_next_state = self.q_a(next_states).detach().max(1)[0]
                self.optimizer = self.optimizer_B

            # Compute the DQN target according to Equation (3) of the paper.
            target = (rewards + self.gamma * max_q_next_state) * (1 - dones.float())

            # Reset gradients
            self.optimizer.zero_grad()

            # Compute the loss and update the weights.
            loss = self.loss_function(q_actual, target)

            # Backpropagate
            loss.backward()
            self.optimizer.step()

            # Sync the weights of q_a and q_b every sync_target steps
            if total_steps % self.sync_target == 0:
                self.sync_weights()

            # Statistics: log the loss to tensorboard
            self.writer.add_scalar("Loss/train", loss.item(), total_steps)

    def sync_weights(self):
        self.q_b.load_state_dict(self.q_a.state_dict())

    def backup_weights(self, path):
        torch.save(self.q_a.state_dict(), path)
