# dqn_agent.py
"""
DQNAgent implementation for CartPole-v1 using PyTorch with:
 - ReplayBuffer: experience replay buffer using collections.deque
 - QNetwork: simple feedforward neural network (2 hidden layers)
 - DQNAgent: agent wrapper with epsilon-greedy action selection, training routine,
   a target network for stable learning, soft/hard updates, and save/load including target network.
"""

import random
from collections import deque, namedtuple
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

Experience = namedtuple("Experience", ("state", "action", "reward", "next_state", "done"))


class ReplayBuffer:
    """Experience Replay Buffer using deque."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states = np.vstack([e.state for e in batch]).astype(np.float32)
        actions = np.array([e.action for e in batch]).astype(np.int64)
        rewards = np.array([e.reward for e in batch]).astype(np.float32)
        next_states = np.vstack([e.next_state for e in batch]).astype(np.float32)
        dones = np.array([e.done for e in batch]).astype(np.uint8)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """Simple Feedforward Q-Network with two hidden layers."""

    def __init__(self, state_size: int, action_size: int, hidden_dim: int = 128):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_size)
        )

    def forward(self, x):
        return self.model(x)


class DQNAgent:
    """DQN Agent implementing experience replay, epsilon-greedy action selection, and a target network."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_size: int = 10000,
        batch_size: int = 64,
        device: Optional[str] = None,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 500,
        target_update_tau: float = 1.0,  # tau=1.0 => hard update
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size

        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Q-networks: policy (online) and target
        self.q_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network = QNetwork(state_size, action_size).to(self.device)
        # Initialize target same as policy
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

        # Epsilon scheduling
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay  # decay rate in episodes

        # target update parameters (soft update with tau or hard update when tau == 1.0)
        self.target_update_tau = target_update_tau

        # Counters
        self.training_steps = 0

        # For reproducibility
        self.seed = None

    def set_seed(self, seed: int):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def select_action(self, state: np.ndarray, epsilon: Optional[float] = None) -> int:
        """
        Selects an action using epsilon-greedy policy.
        If epsilon is None, uses agent.epsilon.
        """
        if epsilon is None:
            epsilon = self.epsilon

        if random.random() < epsilon:
            return random.randrange(self.action_size)
        else:
            state_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_t)
            action = int(q_values.argmax(dim=1).item())
            return action

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def train_step(self):
        """
        Performs a single training step (sample a minibatch and optimize the Q-network).
        Uses the target network to compute stable target Q-values.
        Returns the loss value or None if not enough samples yet.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states_t = torch.from_numpy(states).to(self.device)
        actions_t = torch.from_numpy(actions).unsqueeze(1).to(self.device)  # shape (batch,1)
        rewards_t = torch.from_numpy(rewards).to(self.device)
        next_states_t = torch.from_numpy(next_states).to(self.device)
        dones_t = torch.from_numpy(dones).to(self.device).float()

        # Compute current Q values (policy network)
        curr_q = self.q_network(states_t).gather(1, actions_t).squeeze(1)  # shape (batch,)

        # Compute next Q values from target network
        with torch.no_grad():
            next_q = self.target_network(next_states_t).max(dim=1)[0]
            target_q = rewards_t + (1.0 - dones_t) * (self.gamma * next_q)

        loss = self.criterion(curr_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update training steps counter
        self.training_steps += 1

        return loss.item()

    def decay_epsilon(self, episode: int):
        """
        Exponential decay of epsilon over episodes.
        """
        decay = np.exp(-episode / self.epsilon_decay)
        self.epsilon = max(self.epsilon_end, self.epsilon_start * decay)

    def soft_update_target(self, tau: float):
        """
        Soft update target network parameters:
        target = tau * policy + (1 - tau) * target
        If tau == 1.0, does hard update (copy).
        """
        if tau >= 0.9999:
            self.target_network.load_state_dict(self.q_network.state_dict())
        else:
            for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def update_target(self, hard: bool = True):
        """
        Convenience wrapper to do a hard or soft update using configured tau.
        """
        if hard:
            self.target_network.load_state_dict(self.q_network.state_dict())
        else:
            self.soft_update_target(self.target_update_tau)

    def save(self, path: str):
        """
        Saves the model state dicts and metadata.
        """
        payload = {
            "model_state_dict": self.q_network.state_dict(),
            "target_state_dict": self.target_network.state_dict(),
            "state_size": self.state_size,
            "action_size": self.action_size,
        }
        torch.save(payload, path)

    def load(self, path: str):
        """
        Loads model state dict saved with save(). Leaves target and policy as loaded.
        """
        payload = torch.load(path, map_location=self.device)
        if "state_size" in payload and payload["state_size"] != self.state_size:
            raise ValueError("Saved model state_size doesn't match the agent.")
        if "action_size" in payload and payload["action_size"] != self.action_size:
            raise ValueError("Saved model action_size doesn't match the agent.")
        self.q_network.load_state_dict(payload["model_state_dict"])
        if "target_state_dict" in payload:
            self.target_network.load_state_dict(payload["target_state_dict"])
        else:
            # If no target saved, copy policy
            self.target_network.load_state_dict(self.q_network.state_dict())
        self.q_network.to(self.device)
        self.target_network.to(self.device)
        self.q_network.eval()
        self.target_network.eval()