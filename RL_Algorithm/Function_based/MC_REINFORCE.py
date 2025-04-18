from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base_function import BaseAlgorithm


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
from collections import namedtuple, deque
import random
import matplotlib
import matplotlib.pyplot as plt
import os

class MC_REINFORCE_network(nn.Module):
    """
    Neural network for the MC_REINFORCE algorithm.
    
    Args:
        n_observations (int): Number of input features.
        hidden_size (int): Number of hidden neurons.
        n_actions (int): Number of possible actions.
        dropout (float): Dropout rate for regularization.
    """

    def __init__(self, n_observations, hidden_size, n_actions, dropout):
        super(MC_REINFORCE_network, self).__init__()
        # ========= put your code here ========= #
        self.fc1 = nn.Linear(n_observations, hidden_size)
        self.fc3 = nn.Linear(hidden_size, n_actions)
        # ====================================== #

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            Tensor: Output tensor representing action probabilities.
        """
        # ========= put your code here ========= #
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc3(x))
        action_probs = F.softmax(x, dim=-1)
        action_probs = torch.clamp(action_probs, min=1e-8, max=1.0)
        return action_probs
        # ====================================== #

class MC_REINFORCE(BaseAlgorithm):
    def __init__(
            self,
            device = None,
            num_of_action: int = 2,
            action_range: list = [-2.5, 2.5],
            n_observations: int = 4,
            hidden_dim: int = 64,
            dropout: float = 0.5,
            learning_rate: float = 0.01,
            discount_factor: float = 0.95,
    ) -> None:
        """
        Initialize the CartPole Agent.

        Args:
            learning_rate (float): The learning rate for updating Q-values.
            initial_epsilon (float): The initial exploration rate.
            epsilon_decay (float): The rate at which epsilon decays over time.
            final_epsilon (float): The final exploration rate.
            discount_factor (float, optional): The discount factor for future rewards. Defaults to 0.95.
        """     

        # Feel free to add or modify any of the initialized variables above.
        # ========= put your code here ========= #
        self.LR = learning_rate

        self.policy_net = MC_REINFORCE_network(n_observations, hidden_dim, num_of_action, dropout).to(device)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate)

        self.device = device
        self.hidden = hidden_dim
        self.steps_done = 0

        self.episode_durations = []
        
        # ====================================== #

        super(MC_REINFORCE, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
        )
    
    def calculate_stepwise_returns(self, rewards):
        """
        Compute stepwise returns for the trajectory.

        Args:
            rewards (list): List of rewards obtained in the episode.
        
        Returns:
            Tensor: Normalized stepwise returns.
        """
        # ========= put your code here ========= #

        # Monte Carlo Return Calculation
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + self.discount_factor * R
            returns.insert(0, R)

        # Convert to tensor and normalize
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        returns = F.normalize(returns, dim=0)

        return returns
        # ====================================== #

    def generate_trajectory(self, env):
        """
        Generate a trajectory by interacting with the environment.

        Args:
            env: The environment object.
        
        Returns:
            Tuple: (episode_return, stepwise_returns, log_prob_actions, trajectory)
        """
        # ===== Initialize trajectory collection variables ===== #
        # Reset environment to get initial state (tensor)
        # Store state-action-reward history (list)
        # Store log probabilities of actions (list)
        # Store rewards at each step (list)
        # Track total episode return (float)
        # Flag to indicate episode termination (boolean)
        # Step counter (int)
        # ========= put your code here ========= #

        # Initialization
        state, _ = env.reset()

        trajectory = []
        log_prob_actions = []
        rewards = []
        entropy_list = []

        done = False
        timestep = 0
        episode_return = 0.0
        # ====================================== #
        
        # Trajectory Collection Loop
        while not done:
            
            # Get state tensor and predict action
            state_tensor = torch.tensor([state['policy'][0, i] for i in range(4)], dtype=torch.float32).to(self.device)
            action_probs = self.policy_net(state_tensor)
            dist = distributions.Categorical(action_probs)
            action = dist.sample()

            # Get log-probability and entropy of action
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            entropy_list.append(entropy)

            # Scale action and step environment
            action = action.view(1, -1)
            action = self.scale_action(action)            
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Save log-prob, reward, and full transition
            log_prob_actions.append(log_prob)
            rewards.append(reward)
            trajectory.append((state_tensor, action.item(), reward))

            # Update state and counters
            state = next_state
            episode_return += reward
            timestep += 1
            done = terminated or truncated

        # Prepare return values
        stepwise_returns = self.calculate_stepwise_returns(rewards)
        log_prob_actions = torch.stack(log_prob_actions)
        return timestep, episode_return, stepwise_returns, log_prob_actions, trajectory, torch.stack(entropy_list)
        # ====================================== #
    
    def calculate_loss(self, stepwise_returns, log_prob_actions, entropy_list):
        """
        Compute the loss for policy optimization.

        Args:
            stepwise_returns (Tensor): Stepwise returns for the trajectory.
            log_prob_actions (Tensor): Log probabilities of actions taken.
        
        Returns:
            Tensor: Computed loss.
        """
        # ========= put your code here ========= #
        # Policy Gradient Loss
        return -torch.sum(log_prob_actions * stepwise_returns)/len(stepwise_returns)
        # ====================================== #

    def update_policy(self, stepwise_returns, log_prob_actions, entropy_list):
        """
        Update the policy using the calculated loss.

        Args:
            stepwise_returns (Tensor): Stepwise returns.
            log_prob_actions (Tensor): Log probabilities of actions taken.
        
        Returns:
            float: Loss value after the update.
        """
        # ========= put your code here ========= #
        # Calculate loss
        loss = self.calculate_loss(stepwise_returns, log_prob_actions, entropy_list)

        # Backward pass and optimizer step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
        # ====================================== #
    
    def learn(self, env):
        """
        Train the agent on a single episode.

        Args:
            env: The environment to train in.
        
        Returns:
            Tuple: (episode_return, loss, trajectory)
        """
        # ========= put your code here ========= #

        # Train policy for 1 episode
        self.policy_net.train()
        timestep, episode_return, stepwise_returns, log_prob_actions, trajectory, entropy = self.generate_trajectory(env)
        
        # Policy update
        loss = self.update_policy(stepwise_returns, log_prob_actions, entropy)
        
        # Return training info
        return timestep, episode_return, stepwise_returns, loss, trajectory
        # ====================================== #

    def save_w(self, path, filename):
        """
        Save model weight.
        """
        # ========= put your code here ========= #
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, filename)
        torch.save(self.policy_net.state_dict(), file_path)
        print(f"[INFO] Saved model weights to {file_path}")
        # ====================================== #
            
    def load_w(self, path, filename):
        """
        Load model weight.
        """
        # ========= put your code here ========= #
        file_path = os.path.join(path, filename)
        if os.path.exists(file_path):
            self.policy_net.load_state_dict(torch.load(file_path, map_location=self.device))
            print(f"[INFO] Loaded model weights from {file_path}")
        else:
            raise FileNotFoundError(f"[ERROR] File not found: {file_path}")
        # ====================================== #
