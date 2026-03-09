from abc import ABC, abstractmethod
import torch

from agent.learner import Learner
import settings

class QLearner(Learner):
    def __init__(self, agent):
        super().__init__(agent)

    def get_loss(self, observation, next_observation, reward, terminated):
        if terminated:
            y = reward
        else:
            y = reward + settings.DISCOUNT * self.agent.target_evaluate(next_observation)
        
        return (y - self.agent.current_evaluate(observation)) ** 2

    def get_loss_batch(self, samples_batch):
        observations = torch.FloatTensor([sample[0] for sample in samples_batch])
        actions = torch.IntTensor([sample[1] for sample in samples_batch])
        rewards = torch.FloatTensor([sample[2] for sample in samples_batch])
        next_observations = torch.FloatTensor([sample[3] for sample in samples_batch])
        episode_ends =  torch.BoolTensor([sample[4] for sample in samples_batch])

        current_q_values = self.agent.current_evaluate(observations)
        next_q_values = self.agent.target_evaluate(next_observations)
        
        selected_q_values = current_q_values[torch.arange(settings.BATCH_SIZE), actions]
        non_terminal_mask = ~episode_ends
        targets = rewards.clone()

        if torch.any(non_terminal_mask):
            max_next_q_values = torch.max(next_q_values[non_terminal_mask], dim=1)[0]
            targets[non_terminal_mask] += settings.DISCOUNT * max_next_q_values
        
        loss_batch = (targets - selected_q_values)**2
        return loss_batch.mean()
