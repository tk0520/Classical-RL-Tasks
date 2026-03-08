from agent.learning import get_loss, get_loss_batch
import time

import settings

def train(env, agent):
    step_count = 0

    for episode_count in range(settings.NUM_EPISODES):
        observation, _ = env.reset()
        total_reward = 0 
        episode_end = False

        while not episode_end:
            action = agent.act(observation)

            next_observation, reward, terminated, truncated, _ = env.step(action)
            episode_end = terminated or truncated
            experience = (observation, action, reward, next_observation, episode_end)

            agent.replay_memory.store(experience)
            observation = next_observation
            step_count += 1
            total_reward += reward
            time.sleep(0.000005)

            if len(agent.replay_memory) > settings.INITIAL_MEMORY:
                samples_batch = agent.replay_memory.get_samples()
                loss_batch = get_loss_batch(samples_batch, agent)
                agent.optimizer.zero_grad()

                loss_batch.backward()
                agent.optimizer.step()
            
            if step_count > settings.EPSILON_DECAY_COUNT:
                agent.epsilon_decay()
            
            if step_count % settings.UPDATE_INTERVAL == 0:
                agent.update_target_model()
                print("Target Model Updated!")

            if episode_end:
                print("step_count:", step_count)
                print("Episode Count:", episode_count)
                print("Episode Reward:", total_reward)
        