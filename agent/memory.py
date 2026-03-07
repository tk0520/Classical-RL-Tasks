from collections import deque
import random

import settings

class ReplayMemory:
    def __init__(self):
        self.memory = deque(maxlen=settings.MAX_MEMORY)
    
    def store(self, experience):
        self.memory.append(experience)
    
    def get_samples(self):
        return random.sample(self.memory, settings.BATCH_SIZE)
    
    def __len__(self):
        return len(self.memory)