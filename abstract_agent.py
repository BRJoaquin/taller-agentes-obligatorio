import torch
import torch.nn as nn
import numpy as np
from replay_memory import ReplayMemory
from abc import ABC, abstractmethod
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from mario_utils import show_video


class Agent(ABC):
    def __init__(
        self,
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
        steps_before_training=10000,
    ):
        # Assign device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Transform the observation into a tensor
        self.state_processing_function = obs_processing_func

        # Create the replay memory
        self.memory = ReplayMemory(memory_buffer_size, self.device)

        self.env = gym_env

        # Hyperparameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_i
        self.epsilon_i = epsilon_i
        self.epsilon_f = epsilon_f
        self.epsilon_anneal = epsilon_anneal_time
        self.epsilon_decay = epsilon_decay
        self.episode_block = episode_block
        self.steps_before_training = steps_before_training

    def train(
        self,
        number_episodes=50000,
        max_steps_episode=10000,
        max_steps=1000000,
        writer_name="default_writer_name",
    ):
        # Save the rewards and steps of each episode to statistics
        rewards = []
        episodes_steps = []

        # Initialize the total number of steps of the training
        total_steps = 0

        # Create the writer for tensorboard
        self.writer = SummaryWriter(comment="-" + writer_name)

        for ep in tqdm(range(number_episodes), unit=" episodes"):
            # Stop the training if we reach the maximum number of steps
            if total_steps > max_steps:
                break

            # Reset the environment
            state = self.env.reset()
            x_pos, time = self.get_initial_info(self.env)

            current_episode_reward = 0.0
            episode_steps = 0

            for s in range(max_steps):
                # select the next action
                action = self.select_action(state, total_steps, True, x_pos, time)

                # Execute the action and get the next state, reward and done flag
                next_state, reward, done, info = self.env.step(action)

                current_episode_reward += reward
                total_steps += 1
                episode_steps += 1
                next_x_pos = info["x_pos"]
                next_time = info["time"]

                # Save the transition in memory
                self.add_to_memory(
                    state,
                    action,
                    reward,
                    done,
                    next_state,
                    x_pos,
                    time,
                    next_x_pos,
                    next_time,
                )

                # Move to the next state
                state = next_state
                x_pos = next_x_pos
                time = next_time

                # Update the weights of the network
                self.update_weights(total_steps)

                # We don't want to play forever (a way to truncate the episode)
                if done or episode_steps > max_steps_episode:
                    break

            # Append the rewards and steps of the episode
            rewards.append(current_episode_reward)
            episodes_steps.append(episode_steps)

            # Save the model every 1000 episodes (just in case)
            if ep % 1000 == 0:
                self.backup_weights(f"backup/model_{ep}.pt")

            # Statistics
            mean_reward = np.mean(rewards[-100:])
            mean_steps = np.mean(episodes_steps[-100:])
            self.writer.add_scalar("epsilon", self.epsilon, total_steps)
            self.writer.add_scalar("reward_100", mean_reward, total_steps)
            self.writer.add_scalar("steps_100", mean_steps, total_steps)
            self.writer.add_scalar("reward", current_episode_reward, total_steps)

            # Report on the traning rewards every EPISODE BLOCK episodes
            if ep % self.episode_block == 0:
                print(
                    f"Episode {ep} - Avg. Reward over the last {self.episode_block} episodes {np.mean(rewards[-self.episode_block:])} epsilon {self.epsilon} steps {np.mean(episodes_steps[-self.episode_block:])} total steps {total_steps}"
                )
        # Report when the training is finished
        print(
            f"Episode {ep + 1} - Avg. Reward over the last {self.episode_block} episodes {np.mean(rewards[-self.episode_block:])} epsilon {self.epsilon} total steps {total_steps}"
        )

        self.writer.close()

        return rewards

    def save_model(self, policy, path):
        torch.save(policy.state_dict(), path)

    # Calculate the epsilon value for the current step
    # If epsilon_anneal is not set, then the epsilon value will be decayed exponentially
    def compute_epsilon(self, steps_so_far):
        if steps_so_far < self.steps_before_training:
            return self.epsilon_i

        traing_steps = steps_so_far - self.steps_before_training
        if hasattr(self, "epsilon_anneal") and self.epsilon_anneal is not None:
            if traing_steps > self.epsilon_anneal:
                self.epsilon = self.epsilon_f
            else:
                self.epsilon = self.epsilon_i - (self.epsilon_i - self.epsilon_f) * (
                    traing_steps / self.epsilon_anneal
                )
        else:
            self.epsilon = max(
                self.epsilon_f, self.epsilon_i * (self.epsilon_decay**traing_steps)
            )
        return self.epsilon

    def record_test_episode(self, env):
        done = False

        # Observar estado inicial como indica el algoritmo
        state = env.reset()
        x_pos, time = self.get_initial_info(env)

        step_counter = 0
        while not done:
            env.render()  # Queremos hacer render para obtener un video al final.

            # Seleccione una accion de forma completamente greedy.
            action = self.select_action(
                state, step_counter, train=False, x_pos=x_pos, time=time
            )

            # Ejecutar la accion, observar resultado y procesarlo como indica el algoritmo.
            next_state, reward, done, info = env.step(action)
            next_x_pos = info["x_pos"]
            next_time = info["time"]

            # print(f"Step {step_counter} - Action {action} - Reward {reward} - Info {info}")

            step_counter += 1

            if done:
                break

            # Actualizar el estado
            state = next_state
            x_pos = next_x_pos
            time = next_time

        env.close()
        show_video()

    # Get the initial information of the environment
    def get_initial_info(self, env):
        # not a fancy way to get the initial information (take the first step and do nothing)
        _, _, _, info = env.step(0)
        return info["x_pos"], info["time"]

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
        self.memory.add(state, action, reward, done, next_state)

    @abstractmethod
    def select_action(self, state, current_steps, train=True, x_pos=None, time=None):
        pass

    @abstractmethod
    def update_weights(self, total_steps):
        pass

    @abstractmethod
    def sync_weights(self):
        pass

    @abstractmethod
    def backup_weights(self, path):
        pass
