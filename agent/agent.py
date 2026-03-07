from agent.model import DQN
from torch.optim import Adam
import torch

from agent.memory import ReplayMemory
import settings

class CartPoleAgent:
    def __init__(self, num_action_space):
        self.current_model = DQN(num_action_space)
        self.target_model = DQN(num_action_space)
        self.optimizer = Adam(self.current_model.parameters(), lr=settings.LEARNING_RATE)
        self.replay_memory = ReplayMemory()
        
        self.target_model.load_state_dict(self.current_model.state_dict())

    def current_evaluate(self, observation):
        return self.current_model.forward(observation).max()
    
    def target_evaluate(self, observation):
        with torch.no_grad():
            return self.target_model.forward(observation).max()

    def act(self, observation):
        q_values = self.current_model.forward(observation)
        return torch.argmax(q_values).item()

