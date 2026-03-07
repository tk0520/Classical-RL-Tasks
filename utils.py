from agent.learning import get_loss
import time

import settings

def train(env, agent):
    for episode_count in range(settings.NUM_EPISODES):
        observation, _ = env.reset()
        
        step_count = 0
        episode_end = False

        while not episode_end:
            action = agent.act(observation)

            next_observation, reward, terminated, truncated, _ = env.step(action)
            experience = (observation, action, reward, next_observation)
            episode_end = terminated or truncated

            agent.replay_memory.store(experience)
            observation = next_observation
            step_count += 1
            time.sleep(0.005)

            if len(agent.replay_memory) > settings.INITIAL_MEMORY:
                print("samples:", agent.replay_memory.get_samples())
            # loss = get_loss(observation, next_observation, reward, episode_end, agent)
            # agent.optimizer.zero_grad()

            # loss.backward()
            # agent.optimizer.step()

            if episode_end:
                print("step_count:", step_count)
                print("Episode Count:", episode_count)
        