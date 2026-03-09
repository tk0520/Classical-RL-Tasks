from gymnasium.wrappers import RecordVideo
import gymnasium as gym

from agent.agent import CartPoleAgent
from utils import train, train_with_trajectory
import settings

try:
    env = gym.make("CartPole-v1", render_mode=settings.RENDER_MODE)
    env = RecordVideo(
        env,
        video_folder="videos",
        episode_trigger=lambda episode_id: episode_id % settings.VIDEO_INTERVAL == 0
    )
    agent = CartPoleAgent(env.observation_space.shape[0], env.action_space.n)
    # train(env, agent)
    train_with_trajectory(env, agent)
finally:
    env.close()