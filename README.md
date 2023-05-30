# Abstract 

This project applies Deep Reinforcement Learning (DRL) techniques to train an agent to play Super Mario Bros using OpenAI's Gym library and the NES emulator environment.

# Problem

The task for our agent is to navigate through a series of levels in the classic Super Mario Bros game. This is a complex task involving both short-term and long-term decision making. The agent needs to avoid enemies and obstacles, collect rewards, and also understand the structure of the game levels to successfully navigate through them. The agent has to make these decisions based only on the visual input from the game screen.

# Solution

The solution uses a variant of Q-Learning known as Deep Q-Learning (DQN), where a deep neural network is used to approximate the Q-function. The DQN agent interacts with the environment, stores the transition information in its memory, and uses this information to update the weights of the neural network.

The architecture of the network is as follows:

- The input to the network is a stack of four consecutive grayscale game frames. This allows the network to understand the motion of game objects.
- These frames pass through three convolutional layers, which extract visual features from the game screen.
- The output of the convolutional layers is flattened and passed through a fully connected layer.
- The final layer of the network produces the Q-values for each possible action.

Key techniques used in the DQN algorithm include:

- **Experience Replay**: The agent stores its experiences in a memory buffer and later samples a random batch of experiences to update the network weights. This helps to break correlations in the observation sequence and stabilizes learning.
- **Epsilon-Greedy Strategy**: The agent uses an epsilon-greedy strategy for exploration, where it chooses a random action with probability epsilon, and chooses the action with the highest predicted Q-value otherwise. This helps the agent to explore the game environment and learn better policies.
- **Target Network**: The DQN algorithm uses a separate target network to compute the target Q-values during learning. The weights of the target network are updated periodically from the policy network. This helps to stabilize learning.

The code also includes various utilities for preprocessing game frames and recording the agent's performance.

# Training Procedure and Hyperparameters

Training of the agent is accomplished by interacting with the game environment over a series of episodes. In each episode, the agent selects actions according to an epsilon-greedy strategy, observes the reward and the next state, and stores this information in a replay memory.

At each time step, if the replay memory has enough experiences stored, a mini-batch of experiences is randomly sampled and used to perform a learning update on the policy network.

The agent's exploration rate (epsilon) is initially set to a high value and then gradually annealed towards a lower value throughout training. This encourages the agent to initially explore the environment and later exploit its learned knowledge.

# Evaluation

The performance of the agent can be evaluated by running the trained agent on the game environment and observing its performance. The evaluation phase differs from training in that actions are always selected based on the policy learned by the agent (i.e., the action with the highest predicted Q-value is selected), and no further learning updates are performed.

The agent's performance is recorded as a video, which can be visualized for a more intuitive understanding of the agent's behavior.
