import gymnasium as gym
import time

from agent.agent import CartPoleAgent
from agent.learning import get_loss
from utils import train
import settings

try:
    env = gym.make("CartPole-v1", render_mode=settings.RENDER_MODE)
    agent = CartPoleAgent(env.action_space.n)
    train(env, agent)

finally:
    env.close()