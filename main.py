from gymnasium.wrappers import RecordVideo
import gymnasium as gym

from agent.agent import CartPoleAgent
from utils import train
import settings

try:
    env = gym.make("CartPole-v1", render_mode=settings.RENDER_MODE)
    env = RecordVideo(
        env,
        video_folder="videos",
        episode_trigger=lambda episode_id: episode_id % settings.VIDEO_INTERVAL == 0
    )
    
    agent = CartPoleAgent(env.action_space.n)
    train(env, agent)
finally:
    env.close()