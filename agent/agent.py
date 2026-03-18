from torch.optim import Adam
import numpy as np
import random
import torch

from agent.learner.q_learning import QLearner, NStepQLearner
from agent.memory import ReplayMemory
from agent.model import DQN, DQNAtari
import settings

class CartPoleAgent:
    def __init__(self, num_state_space, num_action_space):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.current_model = DQN(num_state_space, num_action_space).to(device=self.device)
        self.target_model = DQN(num_state_space, num_action_space).to(device=self.device)
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
            action = random.choice(self.actions)
            return action
        
        with torch.no_grad():
            if isinstance(observation, np.ndarray):
                observation = torch.from_numpy(observation).float()

            if observation.ndim == 3:
                observation = observation.unsqueeze(0)
            
            device = next(self.current_model.parameters()).device
            observation = observation.to(device)
    
            q_values = self.current_model.forward(observation)
            return q_values.squeeze(0).argmax().item()

class AtariAgent(CartPoleAgent):
    def __init__(self, num_state_space, num_action_space):
        super().__init__(num_state_space, num_action_space)
        self.current_model = DQNAtari(num_state_space, num_action_space).to(device=self.device)
        self.target_model = DQNAtari(num_state_space, num_action_space).to(device=self.device)