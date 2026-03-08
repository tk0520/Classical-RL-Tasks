from agent.learning import get_loss, get_loss_batch
import matplotlib.pyplot as plt

import settings

def record_stat(total_rewards):
    plt.plot(total_rewards, color=settings.PLOT_COLOR, linewidth=1)
    plt.xlabel("Episodes")
    plt.ylabel("Total Rewards")
    plt.savefig(f"./stats/{len(total_rewards)}_graph.png")
    
def train(env, agent):
    step_count = 0
    total_rewards = []

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

            if len(agent.replay_memory) > settings.INITIAL_MEMORY:
                samples_batch = agent.replay_memory.get_samples()
                loss_batch = get_loss_batch(samples_batch, agent)
                agent.optimizer.zero_grad()

                loss_batch.backward()
                agent.optimizer.step()
            
            if step_count > settings.EPSILON_DECAY_COUNT:
                agent.epsilon_decay()
            
            if episode_count % settings.UPDATE_INTERVAL == 0:
                agent.update_target_model()
                
            if episode_count % settings.STAT_INTERVAL == 0:
                record_stat(total_rewards)

            if episode_end:
                total_rewards.append(total_reward)
                print("step_count:", step_count)
                print("Episode Count:", episode_count)
                print("Episode Reward:", total_reward)
        