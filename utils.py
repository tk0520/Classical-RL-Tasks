from agent.learning import get_loss, get_loss_batch
import matplotlib.pyplot as plt
import torch
import copy

import settings

def record_stat(total_rewards):
    total_rewards_copy = copy.deepcopy(total_rewards)
    total_rewards_per_50 = total_rewards_copy[::50]

    plt.plot(total_rewards_per_50, color=settings.PLOT_COLOR, linewidth=1)
    plt.xlabel("Episodes")
    plt.ylabel("Total Rewards per 50 Episodes")
    plt.savefig(f"./stats/{len(total_rewards)}_graph.png")
    plt.close()
    
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
                loss_batch = agent.learner.get_loss_batch(samples_batch)
                agent.optimizer.zero_grad()

                loss_batch.backward()
                agent.optimizer.step()
            
            if step_count > settings.EPSILON_DECAY_COUNT:
                agent.epsilon_decay()
            
            if episode_count % settings.UPDATE_INTERVAL == 0:
                agent.update_target_model()
                
            if episode_count % settings.STAT_INTERVAL == 0:
                record_stat(total_rewards)
            
            if episode_count % settings.MODEL_SAVE_INTERVAL == 0:
                torch.save(agent.current_model, f"models/DQN_CartPole_{episode_count}.pt")
        
            if episode_end:
                total_rewards.append(total_reward)
                print("step_count:", step_count)
                print("Episode Count:", episode_count)
                print("Episode Reward:", total_reward)
