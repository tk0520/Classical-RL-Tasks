import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch

class DQN(nn.Module):
    def __init__(self, num_state_space, num_action_space):
        super().__init__()
        self.dense1 = nn.Linear(num_state_space, num_state_space * 4)
        output_size = num_state_space * 4
        self.dense2 = nn.Linear(output_size, output_size * 4)
        output_size *= 4
        self.dense3 = nn.Linear(output_size, output_size * 4)
        output_size *= 4
        self.output = nn.Linear(output_size, num_action_space)
    
    def preprocess(self, x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        return x
        
    def forward(self, x):
        x = self.preprocess(x)

        x = self.dense1(x)
        x = F.relu(x)
        
        x = self.dense2(x)
        x = F.relu(x)

        x = self.dense3(x)
        x = F.relu(x)

        x = self.output(x)
        return x

class DQNAtari(nn.Module):
    def __init__(self, num_state_space, num_action_space):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=num_state_space, out_channels=16,
            kernel_size=(8,8), stride=4
        )
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32,
            kernel_size=(4,4), stride=2
        )
        self.dense1 = nn.LazyLinear(256)
        self.dense2 = nn.LazyLinear(128)
        self.dense_output = nn.Linear(128, num_action_space)
    
    def preprocess(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        else:
            x = x.float()

        device = next(self.parameters()).device
        return x.to(device)
        
    def forward(self, x):
        x = self.preprocess(x)

        x = self.conv1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = F.relu(x)

        x = x.view(x.size(0), -1)

        x = self.dense1(x)
        x = F.relu(x)

        x = self.dense2(x)
        x = F.relu(x)

        x = self.dense_output(x)
        return x