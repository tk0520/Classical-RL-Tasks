from gymnasium.wrappers import RecordVideo, AtariPreprocessing, FrameStackObservation
from gymnasium.wrappers.vector import RecordEpisodeStatistics
import gymnasium as gym
import ale_py

from agent.agent import CartPoleAgent, AtariAgent
from utils import train, train_with_trajectory
import settings

try:
    gym.register_envs(ale_py)
    env = gym.make("ALE/Breakout-v5", render_mode=settings.RENDER_MODE)

    env = AtariPreprocessing(
        env, 
        screen_size=84,        # 84x84로 리사이즈
        grayscale_obs=True,     # 흑백 변환
        frame_skip=1,           # 4프레임 스킵        
        scale_obs=True,          # 픽셀값 0~1로 정규화
        terminal_on_life_loss=False
    )
    env = FrameStackObservation(env, stack_size=4)
    env = RecordVideo(
        env,
        video_folder="videos",
        episode_trigger=lambda episode_id: episode_id % settings.VIDEO_INTERVAL == 0
    )

    agent = AtariAgent(env.observation_space.shape[0], env.action_space.n)
    train(env, agent)
finally:
    env.close()