import torch
import torch.nn as nn
import torch.nn.functional as F
from replay_memory import ReplayMemory
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
import numpy as np
from abstract_agent import Agent


class DQNAgent(Agent):
    def __init__(
        self,
        gym_env,
        model,
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
        # Assign policy net to device
        self.policy_net = model.to(self.device)
        # Assign target net to device
        self.target_net = model.to(self.device)

        # Assign loss function (MSE) to device
        self.loss_function = nn.MSELoss().to(self.device)

        # Assign optimizer to policy net parameters
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=learning_rate
        )

    def greedy_policy(self, state):
        # Get the action from the policy network
        # transform the state into a tensor first
        state_tensor = self.state_processing_function(state).to(self.device)
        action = self.policy_net(state_tensor).argmax().item()
        return action

    # Epsilon greedy strategy
    # In case we are training, we select a random action with probability epsilon
    # In case we are not training, we always select the best action (greedy)
    def select_action(self, state, current_steps, train=True, x_pos=None, time=None):
        if not train:
            action = self.greedy_policy(state)
        else:
            epsilon = self.compute_epsilon(current_steps)
            if (
                current_steps < self.steps_before_training
                or np.random.random() < epsilon
            ):
                action = self.env.action_space.sample()
            else:
                action = self.greedy_policy(state)
        return action

    def update_weights(self, total_steps):
        if len(self.memory) > self.batch_size:
            # Reset gradients
            self.optimizer.zero_grad()

            # Sample a batch of transitions from the replay memory
            states, actions, rewards, dones, next_states = self.memory.sample(
                self.batch_size
            )

            # Add a dimension to actions to be able to use gather
            actions = actions.unsqueeze(-1)
            # Get the current Q values for all actions from the policy net
            q_actual = self.policy_net(states).gather(1, actions)

            # Get the max Q values for all actions from the target net
            # We detach the target values from the computational graph to avoid backpropagating through them
            # We don't want to update the target net parameters, we want to update the policy net parameters
            max_q_next_state = self.target_net(next_states).detach().max(1)[0]

            # Compute the target Q values
            # In case the episode is done, the target Q value is zero
            target = (rewards + self.gamma * max_q_next_state) * (1 - dones.float())

            # Compute the loss between actual and target Q values
            loss = self.loss_function(q_actual.squeeze(), target)

            # Backpropagate the loss
            loss.backward()
            self.optimizer.step()

            # Update the target net weights every 100 steps
            if total_steps % 100 == 0:
                self.sync_weights()

            # Statistics: log the loss to tensorboard
            self.writer.add_scalar("Loss/train", loss.item(), total_steps)

    def sync_weights(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def backup_weights(self, path):
        self.policy_net.save(path)
