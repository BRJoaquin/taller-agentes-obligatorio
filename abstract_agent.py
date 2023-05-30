import torch
import torch.nn as nn
import numpy as np
from replay_memory import ReplayMemory, Transition
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
        self.memory = ReplayMemory(memory_buffer_size)

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

        self.total_steps = 0

    def train(
        self,
        number_episodes=50000,
        max_steps_episode=10000,
        max_steps=1000000,
        writer_name="default_writer_name",
    ):
        rewards = []
        episodes_steps = []
        total_steps = 0
        self.writer = SummaryWriter(comment="-" + writer_name)

        for ep in tqdm(range(number_episodes), unit=" episodes"):
            if total_steps > max_steps:
                break

            state = self.env.reset()

            # Observar estado inicial como indica el algoritmo

            current_episode_reward = 0.0
            episode_steps = 0

            for s in range(max_steps):
                # select the next action
                action = self.select_action(state, total_steps, True)

                # Execute the action and get the next state, reward and done flag
                next_state, reward, done, _ = self.env.step(action)

                current_episode_reward += reward
                total_steps += 1
                episode_steps += 1

                # Save the transition in memory
                self.memory.add(state, action, reward, done, next_state)

                # Move to the next state
                state = next_state

                # Actualizar el modelo
                self.update_weights(total_steps)

                # We don't want to play forever (a way to truncate the episode)
                if done or episode_steps > max_steps_episode:
                    break

            rewards.append(current_episode_reward)
            episodes_steps.append(episode_steps)

            # Update the target network every episode_block episodes
            if ep % self.episode_block == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # Save the model every 1000 episodes (just in case)
            if ep % 1000 == 0:
                self.save_model(f"backup/model_{ep}.pt")

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

        print(
            f"Episode {ep + 1} - Avg. Reward over the last {self.episode_block} episodes {np.mean(rewards[-self.episode_block:])} epsilon {self.epsilon} total steps {total_steps}"
        )

        self.writer.close()

        return rewards

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.policy_net.eval()

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

        step_counter = 0
        while not done:
            env.render()  # Queremos hacer render para obtener un video al final.

            # Seleccione una accion de forma completamente greedy.
            action = self.select_action(state, step_counter, train=False)

            # Ejecutar la accion, observar resultado y procesarlo como indica el algoritmo.
            next_state, reward, done, _ = env.step(action)

            # print(f"Step {step_counter} - Action {action} - Reward {reward}")

            step_counter += 1

            if done:
                break

            # Actualizar el estado
            state = next_state

        env.close()
        show_video()

    @abstractmethod
    def select_action(self, state, current_steps, train=True):
        pass

    @abstractmethod
    def update_weights(self, total_steps):
        pass
