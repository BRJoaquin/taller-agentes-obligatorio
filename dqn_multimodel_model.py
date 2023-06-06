import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN_Multimodel_Model(nn.Module):
    def __init__(self, env_inputs, n_actions, pos_dim, time_dim):
        super(DQN_Multimodel_Model, self).__init__()

        self.input_shape = env_inputs
        self.n_actions = n_actions

        # Position and Time dimensions
        self.pos_dim = pos_dim
        self.time_dim = time_dim

        self.conv1 = nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        conv_width = conv2d_size_out(
            conv2d_size_out(conv2d_size_out(self.input_shape[1], 8, 4), 4, 2), 3, 1
        )
        conv_height = conv2d_size_out(
            conv2d_size_out(conv2d_size_out(self.input_shape[2], 8, 4), 4, 2), 3, 1
        )

        linear_input_size = conv_width * conv_height * 64

        # Modify the input size of the first fully connected layer to include position and time inputs
        self.fc1 = nn.Linear(linear_input_size + pos_dim + time_dim, 512)
        self.fc2 = nn.Linear(512, self.n_actions)

    def forward(self, env_input, pos_input, time_input):
        x = F.leaky_relu(self.conv1(env_input))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor

        # Concatenate position and time inputs
        pos_input = pos_input.unsqueeze(-1)
        time_input = time_input.unsqueeze(-1)
        x = torch.cat((x, pos_input, time_input), dim=1)

        x = F.leaky_relu(self.fc1(x))
        return self.fc2(x)
