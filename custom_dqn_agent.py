import torch
import torch.nn as nn
import torch.nn.functional as F
from prioritize_replay_memory import PrioritizedReplayMemory
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
import numpy as np
from abstract_agent import Agent


class CustomDQNAgent(Agent):
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

        # Create the prioritized replay memory
        self.memory = PrioritizedReplayMemory(memory_buffer_size, self.device)

    def greedy_policy(self, state, x_pos=None, time=None):
        # Convert the state to a tensor and send it to the proper device
        state_tensor = self.state_processing_function(state).to(self.device)
        x_pos_tensor = torch.tensor(x_pos).unsqueeze(0).to(self.device)
        time_tensor = torch.tensor(time).unsqueeze(0).to(self.device)

        # Calculate the Q values from both networks and sum them. We don't need to calculate gradients here, so we use torch.no_grad()
        with torch.no_grad():
            q_values = self.q_a(state_tensor, x_pos_tensor, time_tensor) + self.q_b(
                state_tensor, x_pos_tensor, time_tensor
            )
        return torch.argmax(q_values).item()

    # Epsilon greedy strategy
    # In case we are training, we select a random action with probability epsilon
    # In case we are not training, we always select the best action (greedy)
    def select_action(self, state, current_steps, train=True, x_pos=None, time=None):
        if not train:
            action = self.greedy_policy(state, x_pos, time)
        else:
            epsilon = self.compute_epsilon(current_steps)
            if (
                current_steps < self.steps_before_training
                or np.random.random() < epsilon
            ):
                action = self.env.action_space.sample()
            else:
                action = self.greedy_policy(state, x_pos, time)
        return action

    def update_weights(self, total_steps):
        if len(self.memory) > self.batch_size:
            # Get a minibatch from memory. Resulting in tensors of states, actions, rewards, termination flags, and next states.
            (
                states,
                actions,
                rewards,
                dones,
                next_states,
                x_pos,
                time,
                next_x_pos,
                next_time,
                indices,
                weights,
            ) = self.memory.sample(self.batch_size)

            # Randomly update Q_a or Q_b using the other to calculate the value of the next states.
            if np.random.random() < 0.5:
                q_actual = self.q_a(states, x_pos, time).gather(
                    1, actions.unsqueeze(-1)
                )
                max_q_next_state = (
                    self.q_b(next_states, next_x_pos, next_time).detach().max(1)[0]
                )
                self.optimizer = self.optimizer_A
            else:
                q_actual = self.q_b(states, x_pos, time).gather(
                    1, actions.unsqueeze(-1)
                )
                max_q_next_state = (
                    self.q_a(next_states, next_x_pos, next_time).detach().max(1)[0]
                )
                self.optimizer = self.optimizer_B

            # Compute the DQN target according to Equation (3) of the paper.
            target = (rewards + self.gamma * max_q_next_state) * (1 - dones.float())

            # Reset gradients
            self.optimizer.zero_grad()

            # Compute the loss and update the weights.
            loss = self.loss_function(q_actual.squeeze(), target)

            # Update priorities in memory
            loss_value = loss.detach().cpu().item()
            self.memory.update_priorities(
                indices, loss_value * weights
            )

            # Backpropagate
            loss.backward()
            self.optimizer.step()

            # Sync the weights of q_a and q_b every sync_target steps
            if total_steps % self.sync_target == 0:
                self.sync_weights()

            # Statistics: log the loss to tensorboard
            self.writer.add_scalar("Loss/train", loss.item(), total_steps)

    def sync_weights(self):
        if np.random.random() < 0.5:
            self.q_b.load_state_dict(self.q_a.state_dict())
        else:
            self.q_a.load_state_dict(self.q_b.state_dict())

    def backup_weights(self, path):
        self.q_a.save(path + "_a")
        self.q_b.save(path + "_b")

    def add_to_memory(
        self,
        state,
        action,
        reward,
        done,
        next_state,
        x_pos,
        time,
        next_x_pos,
        next_time,
    ):
        self.memory.add(
            state, action, reward, done, next_state, x_pos, time, next_x_pos, next_time
        )
