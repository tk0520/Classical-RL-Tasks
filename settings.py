# Gymnasium
RENDER_MODE = "human"

# Train Params
NUM_EPISODES = 1000

# Agent Params
INITIAL_MEMORY = 35
MAX_MEMORY = 20000           
UPDATE_INTERVAL = 100 
BATCH_SIZE = 32   

# Action-Value Params
DISCOUNT = 0.99

# Optimizer Params 
LEARNING_RATE = 0.001

# Exploration Params 
EPSILON_START = 1.0 
EPSILON_END = 0.01
EPSILON_DECAY = 0.99
EPSILON_DECAY_COUNT = 500
