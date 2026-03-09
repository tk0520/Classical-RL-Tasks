from agent.learning import get_loss, get_loss_batch
import matplotlib.pyplot as plt
import torch
import copy

import settings

class Trajectory:
    def __init__(self, num_limit):
        self.trajectory_buffer = []
        self.num_limit = num_limit
    
    def collect_trajectory(self, experience):
        if len(self.trajectory_buffer) >= self.num_limit:
            return False
        
        self.trajectory_buffer.append(experience)
        return True
    
    def get_trajectory(self):
        collected = copy.deepcopy(self.trajectory_buffer)
        self.trajectory_buffer.clear()
        return collected
    
    def is_terminated(self, experience):
        _, _, _, _, terminated = experience
        terminated = terminated and (len(self.trajectory_buffer) > self.num_limit)
        
        if not terminated:
            return False
        
        return terminated

    def __len__(self):
        return len(self.trajectory_buffer)

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

def train_with_trajectory(env, agent):
    step_count = 0
    total_rewards = []
    trajectory_collector = Trajectory(15)

    for episode_count in range(settings.NUM_EPISODES):
        observation, _ = env.reset()
        total_reward = 0 
        episode_end = False
        trajectory_collector.trajectory_buffer.clear()

        while not episode_end:
            action = agent.act(observation)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            episode_end = terminated or truncated
            
            experience = (observation, action, reward, next_observation, episode_end)
            trajectory_collector.collect_trajectory(experience)

            observation = next_observation
            step_count += 1
            total_reward += reward

            if len(trajectory_collector) >= trajectory_collector.num_limit or episode_end:
                collected_trajectory = trajectory_collector.get_trajectory()
                agent.replay_memory.store(collected_trajectory)
                trajectory_collector.trajectory_buffer.clear()

            if len(agent.replay_memory) > settings.INITIAL_MEMORY:
                trajectory_batch = agent.replay_memory.get_samples()

                for trajectory in trajectory_batch:
                    loss = agent.learner.get_loss(trajectory)
                    agent.optimizer.zero_grad()

                    loss.backward()
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

