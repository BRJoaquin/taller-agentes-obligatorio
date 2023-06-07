import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN_CNN_Model(nn.Module):
    def __init__(self, env_inputs, n_actions):
        super(DQN_CNN_Model, self).__init__()

        # The input size depends on the environment (number of observations)
        self.input_shape = env_inputs

        # The output size is the number of possible actions
        self.n_actions = n_actions

        # Define the architecture
        self.conv1 = nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate the output dimensions of the convolutions
        # This is needed to know the input size of the linear layer
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        conv_width = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.input_shape[1], 8, 4), 4, 2), 3, 1)
        conv_height = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.input_shape[2], 8, 4), 4, 2), 3, 1)

        linear_input_size = conv_width * conv_height * 64

        # Define the fully connected layers
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, self.n_actions)

    def forward(self, env_input):
        x = F.relu(self.conv1(env_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.state_dict(), path)