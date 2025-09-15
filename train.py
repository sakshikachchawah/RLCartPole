# train.py
"""
Training script for the DQN agent on CartPole-v1 with:
 - target network periodic updates (hard copy every target_update_freq steps)
 - checkpointing (save model every checkpoint_interval episodes)
 - training history saved to 'training_history.npz' (scores and losses)
"""

import os
import time
import json

import gymnasium as gym
import numpy as np
import torch

from dqn_agent import DQNAgent

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def train(
    env_name: str = "CartPole-v1",
    max_episodes: int = 400,
    max_steps_per_episode: int = 500,
    lr: float = 1e-3,
    gamma: float = 0.99,
    buffer_size: int = 10000,
    batch_size: int = 64,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: int = 200,
    target_update_freq_steps: int = 1000,  # update target every N training steps
    target_update_tau: float = 1.0,  # if <1.0 will do soft updates with tau
    checkpoint_interval: int = 50,  # save checkpoint every N episodes
    save_path_latest: str = "dqn_model_latest.pth",
    save_path_best: str = "dqn_model_best.pth",
    history_path: str = "training_history.npz",
    seed: int = 42
):
    # Create environment
    env = gym.make(env_name)
    obs_sample, _ = env.reset(seed=seed)
    state_size = obs_sample.shape[0]
    action_size = env.action_space.n

    print(f"State size: {state_size}, Action size: {action_size}")

    # Initialize agent
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        lr=lr,
        gamma=gamma,
        buffer_size=buffer_size,
        batch_size=batch_size,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        target_update_tau=target_update_tau
    )
    agent.set_seed(seed)

    scores = []
    per_episode_losses = []
    best_avg100 = -np.inf
    global_training_steps = 0

    print_every = max(1, max_episodes // 10)
    start_time = time.time()

    for ep in range(1, max_episodes + 1):
        state, _ = env.reset()
        total_reward = 0.0
        done = False
        step = 0
        episode_losses = []

        while not done and step < max_steps_per_episode:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)
                global_training_steps += 1

            # Periodic target update based on global training steps
            if global_training_steps > 0 and (global_training_steps % target_update_freq_steps == 0):
                # If tau==1.0, this is a hard update, otherwise soft update using tau
                agent.update_target(hard=(agent.target_update_tau >= 0.9999))
            state = next_state
            total_reward += reward
            step += 1

        scores.append(total_reward)
        per_episode_losses.append(np.mean(episode_losses) if len(episode_losses) > 0 else 0.0)

        # Decay epsilon based on episode number
        agent.decay_epsilon(ep)

        # Logging
        if ep % print_every == 0 or ep == 1:
            avg_score_last = np.mean(scores[-print_every:])
            elapsed = time.time() - start_time
            print(f"Episode: {ep}/{max_episodes} | Score: {total_reward:.2f} | "
                  f"Avg(last {print_every}): {avg_score_last:.2f} | Epsilon: {agent.epsilon:.4f} | Time: {elapsed:.1f}s")

        # Check for improved performance and save best model
        if len(scores) >= 100:
            avg_last_100 = np.mean(scores[-100:])
            if avg_last_100 > best_avg100:
                best_avg100 = avg_last_100
                agent.save(save_path_best)
                print(f"New best avg100: {best_avg100:.2f}, saved to {save_path_best}")

            # Optional early stopping if solved
            if avg_last_100 >= 195.0:
                print(f"Environment solved in {ep} episodes! Saving final model.")
                agent.save(save_path_latest)
                break

        # Checkpointing every checkpoint_interval episodes
        if ep % checkpoint_interval == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"dqn_model_ep{ep}.pth")
            agent.save(checkpoint_path)
            # Also save a 'latest' copy
            agent.save(save_path_latest)
            # Save training history snapshot
            np.savez(history_path, scores=np.array(scores), losses=np.array(per_episode_losses))
            print(f"Checkpoint saved at episode {ep} -> {checkpoint_path}")

    # Final save
    agent.save(save_path_latest)
    # Save training history
    np.savez(history_path, scores=np.array(scores), losses=np.array(per_episode_losses))
    print(f"Saved final model to {save_path_latest} and training history to {history_path}")

    env.close()
    return agent, scores, per_episode_losses


if __name__ == "__main__":
    trained_agent, training_scores, training_losses = train()