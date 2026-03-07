import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch

class DQN(nn.Module):
    def __init__(self, num_action_space):
        super().__init__()
        self.dense1 = nn.Linear(4, 8)
        self.dense2 = nn.Linear(8, 32)
        self.dense3 = nn.Linear(32, 64)
        self.output = nn.Linear(64, num_action_space)
    
    def preprocess(self, x):
        if isinstance(x, np.ndarray):
            return torch.tensor(x)
        
    def forward(self, x):
        x = self.preprocess(x)

        x = self.dense1(x)
        x = F.relu(x)
        
        x = self.dense2(x)
        x = F.relu(x)

        x = self.dense3(x)
        x = F.relu(x)

        x = self.output(x)
        return torch.argmax(x).item()