from agent.model import DQN
from torch.optim import Adam
import random
import torch

from agent.learner.q_learning import QLearner
from agent.memory import ReplayMemory
import settings

class CartPoleAgent:
    def __init__(self, num_action_space):
        self.current_model = DQN(num_action_space)
        self.target_model = DQN(num_action_space)
        self.optimizer = Adam(self.current_model.parameters(), lr=settings.LEARNING_RATE)
        self.target_model.load_state_dict(self.current_model.state_dict())
        
        self.replay_memory = ReplayMemory()
        self.learner = QLearner(self)
        
        self.epsilon = settings.EPSILON_START
        self.actions = list(range(num_action_space))

    def current_evaluate(self, observation):
        return self.current_model.forward(observation)
    
    def target_evaluate(self, observation):
        with torch.no_grad():
            return self.target_model.forward(observation)

    def update_target_model(self):
        self.target_model.load_state_dict(self.current_model.state_dict())

    def epsilon_decay(self):
        self.epsilon = max(settings.EPSILON_END, self.epsilon * settings.EPSILON_DECAY)

    def act(self, observation):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        
        with torch.no_grad():
            q_values = self.current_model.forward(observation)
            return torch.argmax(q_values).max().item()

