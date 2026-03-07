import gymnasium as gym
import time

from agent.model import DQN
from agent.learning import q_value
import settings

try:
    env = gym.make("CartPole-v1", render_mode=settings.RENDER_MODE)
    observation, info = env.reset()

    model = DQN(env.action_space.n)
    action = model.forward(observation)
    print("Action:", action)

    observation, reward, terminated, truncated, info = env.step(action)
    print("Observation:", observation)
    print("reward", reward)

    q_value(observation, reward, terminated or truncated, model)

finally:
    env.close()