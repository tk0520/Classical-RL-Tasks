from abc import ABC, abstractmethod

class Learner(ABC):
    def __init__(self, agent):
        self.agent = agent

    @abstractmethod
    def get_loss(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_loss_batch(self, *args, **kwargs):
        pass