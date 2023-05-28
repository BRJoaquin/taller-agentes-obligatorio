import torch
import torch.nn as nn
import torch.nn.functional as F
from replay_memory import ReplayMemory, Transition
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
        # Asignar el modelo al agente (y enviarlo al dispositivo adecuado)
        self.policy_net = model.to(self.device)

        # Asignar una función de costo (MSE)  (y enviarla al dispositivo adecuado)
        self.loss_function = nn.MSELoss().to(self.device)

        # Asignar un optimizador (Adam)
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=learning_rate
        )

    # Epsilon greedy strategy
    def select_action(self, state, current_steps, train=True):
        if not train:
            state_tensor = self.state_processing_function(state).to(self.device)
            action = self.policy_net(state_tensor).argmax().item()
        else:
            epsilon = self.compute_epsilon(current_steps)
            if (
                current_steps < self.steps_before_training
                or np.random.random() < epsilon
            ):
                action = self.env.action_space.sample()
            else:
                # Get the action from the policy network
                # transform the state into a tensor first
                state_tensor = self.state_processing_function(state).to(self.device)
                action = self.policy_net(state_tensor).argmax().item()
        return action

    def update_weights(self, total_steps):
        if len(self.memory) > self.batch_size:
            # Resetear gradientes
            self.optimizer.zero_grad()

            # Obtener un minibatch de la memoria. Resultando en tensores de estados, acciones, recompensas, flags de terminacion y siguentes estados.
            transitions = self.memory.sample(self.batch_size)

            # # Enviar los tensores al dispositivo correspondiente.
            states = torch.tensor([t.state for t in transitions], dtype=torch.float, device=self.device)
            actions = torch.cat(
                [torch.tensor(t.action).view(1) for t in transitions]
            ).to(self.device)
            rewards = torch.cat(
                [torch.tensor(t.reward).view(1) for t in transitions]
            ).to(self.device)
            dones = torch.cat([torch.tensor(t.done).view(1) for t in transitions]).to(
                self.device
            )
            next_states = torch.tensor([t.next_state for t in transitions], dtype=torch.float, device=self.device)

            # Obetener el valor estado-accion (Q) de acuerdo a la policy net para todo elemento (estados) del minibatch.

            actions = actions.unsqueeze(-1)  # Agrega una dimensión extra al final
            q_actual = self.policy_net(states).gather(1, actions)

            # Obtener max a' Q para los siguientes estados (del minibatch). Es importante hacer .detach() al resultado de este computo.
            # Si el estado siguiente es terminal (done) este valor debería ser 0.
            max_q_next_state = self.policy_net(next_states).detach().max(1)[0]

            # Compute el target de DQN de acuerdo a la Ecuacion (3) del paper.
            # Si el estado siguiente es terminal (done) este valor debería ser 0.
            target = (rewards + self.gamma * max_q_next_state) * (1 - dones.float())

            # Compute el costo y actualice los pesos.
            # En Pytorch la funcion de costo se llaman con (predicciones, objetivos) en ese orden.
            loss = self.loss_function(q_actual.squeeze(), target)

            loss.backward()
            self.optimizer.step()
            self.policy_net

            self.writer.add_scalar("Loss/train", loss.item(), total_steps)
