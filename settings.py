# Gymnasium
RENDER_MODE = "rgb_array"

# Train Params
NUM_EPISODES = 10000

# Record Params
STAT_INTERVAL = 100
VIDEO_INTERVAL = 100
PLOT_COLOR = "cornflowerblue"

# Agent Params
INITIAL_MEMORY = 250
MAX_MEMORY = 20000           
UPDATE_INTERVAL = 25
BATCH_SIZE = 64  

# Action-Value Params
DISCOUNT = 0.99

# Optimizer Params 
LEARNING_RATE = 0.001

# Exploration Params 
EPSILON_START = 1.0 
EPSILON_END = 0.01
EPSILON_DECAY = 0.99
EPSILON_DECAY_COUNT = 500
