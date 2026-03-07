import gymnasium as gym
import time

from agent.model import DQN
import settings

try:
    env = gym.make("CartPole-v1", render_mode=settings.RENDER_MODE)
    observation, info = env.reset()

    print("Observation:", observation)
    print("Info:", info)
    print(env.action_space.n)

    model = DQN(env.action_space.n)
    action = model.forward(observation)
    print("Action:", action)
finally:
    env.close()