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