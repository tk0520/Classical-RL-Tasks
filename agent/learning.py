import numpy as np
import torch

import settings

def get_loss(observation, next_observation, reward, termination, agent):
    if termination:
        y = reward
    else:
        y = reward + settings.DISCOUNT * agent.target_evaluate(next_observation)
    
    return (y - agent.current_evaluate(observation)) ** 2

def get_loss_batch(samples_batch, agent):
    observations = np.array([sample[0] for sample in samples_batch], dtype=np.float32)
    actions = np.array([sample[1] for sample in samples_batch], dtype=np.int64)
    rewards = np.array([sample[2] for sample in samples_batch], dtype=np.float32)
    next_observations = np.array([sample[3] for sample in samples_batch], dtype=np.float32)
    episode_ends =  np.array([sample[4] for sample in samples_batch], dtype=bool)

    observations = torch.from_numpy(observations)
    actions = torch.from_numpy(actions)
    rewards = torch.from_numpy(rewards)
    next_observations = torch.from_numpy(next_observations)
    episode_ends = torch.from_numpy(episode_ends)

    device = next(agent.current_model.parameters()).device
    observations = observations.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    next_observations = next_observations.to(device)
    episode_ends = episode_ends.to(device)

    # ================================================================
    print(f"observations device: {observations.device}")
    print(f"actions device: {actions.device}")
    print(f"rewards device: {rewards.device}")
    print(f"next_observations device: {next_observations.device}")
    print(f"episode_ends device: {episode_ends.device}")

    current_q_values = agent.current_evaluate(observations)
    next_q_values = agent.target_evaluate(next_observations)


    print(f"current_q_values device: {current_q_values.device}")
    print(f"next_q_values device: {next_q_values.device}")
    # ================================================================
    
    selected_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    non_terminal_mask = ~episode_ends
    targets = rewards.clone()

    if torch.any(non_terminal_mask):
        max_next_q_values = torch.max(next_q_values[non_terminal_mask], dim=1)[0]
        max_next_q_values = max_next_q_values.to(device)

        targets[non_terminal_mask] += settings.DISCOUNT * max_next_q_values
    
    loss_batch = (targets - selected_q_values)**2
    return loss_batch.mean()