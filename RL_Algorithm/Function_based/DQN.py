from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base_function import BaseAlgorithm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import random
import matplotlib
import matplotlib.pyplot as plt
import os

class DQN_network(nn.Module):
    """
    Neural network model for the Deep Q-Network algorithm.
    
    Args:
        n_observations (int): Number of input features.
        hidden_size (int): Number of hidden neurons.
        n_actions (int): Number of possible actions.
        dropout (float): Dropout rate for regularization.
    """
    def __init__(self, n_observations, hidden_size, n_actions, dropout):
        super(DQN_network, self).__init__()

        # === Define layers ===
        self.fc1 = nn.Linear(n_observations, hidden_size)  # Input → Hidden
        self.dropout = nn.Dropout(dropout)                 # Dropout for regularization
        self.fc2 = nn.Linear(hidden_size, hidden_size)     # Hidden → Hidden
        self.out = nn.Linear(hidden_size, n_actions)       # Hidden → Output (Q-values)

    def init_weights(self):
        """
        Initialize network weights using Xavier initialization for better convergence.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # Xavier initialization
                nn.init.zeros_(m.bias)  # Initialize bias to 0

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (Tensor): Input state tensor.
        
        Returns:
            Tensor: Q-value estimates for each action.
        """
        # ========= put your code here ========= #
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.out(x) # Output Q(s, a) for all actions
        # ====================================== #

class DQN(BaseAlgorithm):
    def __init__(
            self,
            device = None,
            num_of_action: int = 11,
            action_range: list = [-12.0, 12.0],
            n_observations: int = 4,
            hidden_dim: int = 128,
            dropout: float = 0.1,
            learning_rate: float = 0.001,
            tau: float = 0.005,
            initial_epsilon: float = 1.0,
            epsilon_decay: float = 1e-3,
            final_epsilon: float = 0.05,
            discount_factor: float = 0.95,
            buffer_size: int = 10000,
            batch_size: int = 32
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
        


        # === Set up networks ===
        self.policy_net = DQN_network(n_observations, self.hidden_dim, self.num_of_action, dropout).to(device)
        self.target_net = DQN_network(n_observations, self.hidden_dim, self.num_of_action, dropout).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # === Set device and hyperparameters ===
        self.device = device
        self.tau = tau
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.learning_rate, amsgrad=True)
        self.episode_durations = []
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.initial_epsilon = initial_epsilon
        self.state_stats = deque(maxlen=1000)
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.num_of_action = num_of_action        

        super(DQN, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,  
            discount_factor=discount_factor,
            buffer_size=buffer_size,
            batch_size=batch_size
        )

    def select_action(self, state):
        """
        Select an action based on an epsilon-greedy policy.
        
        Args:
            state (Tensor): The current state of the environment.
        
        Returns:
            Tensor: The selected action.
        """
        # ========= put your code here ========= #

        if torch.rand(1).item() < self.epsilon:
            # random action index [0, num_of_action-1] shape: [1, 1]
            return torch.tensor([[random.randrange(self.num_of_action)]], device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                q_values = self.policy_net(state)  # shape: [1, num_actions]
                action_idx = torch.argmax(q_values, dim=0, keepdim=True)  # shape: [1, 1]
            return action_idx

        # ====================================== #

    def generate_sample(self, batch_size):
        """
        Generates a batch sample from memory for training.

        Returns:
            Tuple: A tuple containing:
                - non_final_mask (Tensor): A boolean mask indicating which states are non-final.
                - non_final_next_states (Tensor): The next states that are not terminal.
                - state_batch (Tensor): The batch of current states.
                - action_batch (Tensor): The batch of actions taken.
                - reward_batch (Tensor): The batch of rewards received.
        """
        # Ensure there are enough samples in memory before proceeding

        # Sample a batch from memory
        # ========= put your code here ========= #
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample()
        if len(self.memory) < batch_size:
            return None

        # Create mask True where next_state is NOT terminalfor non-final (non-terminal) states
        non_final_mask = ~done_batch  # done = False → non_final
        non_final_next_states = next_state_batch[non_final_mask]

        return non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch

        # ====================================== #

    def calculate_loss(self,
                       non_final_mask, 
                       non_final_next_states, 
                       state_batch, 
                       action_batch, 
                       reward_batch):
        """
        Computes the loss for policy optimization.

        Args:
            non_final_mask (Tensor): Mask indicating which states are non-final.
            non_final_next_states (Tensor): The next states that are not terminal.
            state_batch (Tensor): Batch of current states.
            action_batch (Tensor): Batch of actions taken.
            reward_batch (Tensor): Batch of received rewards.
        
        Returns:
            Tensor: Computed loss.

        This function:
        - Predicts Q(s, a) using the policy network.
        - Computes max Q(s', a') from the target network for non-terminal next states.
        - Calculates the target Q-values using the Bellman equation: 
            target = r + y * max_a' Q_target(s', a')
        - Computes the mean squared error (MSE) loss between predicted and target Q-values.
        """
        # ========= put your code here ========= #

        # Reshape action_batch to match gather's requirements and ensure long type for indexing
        action_batch = action_batch.view(1, -1).long()

        dones = non_final_mask.float()

        # Compute Q(s, a) from the policy network by selecting the Q-values for the taken actions
        q_values = self.policy_net(state_batch).gather(1, action_batch).view(-1)

        # Initialize a tensor to hold Q(s', a') values for each sample in the batch
        next_state_values = torch.zeros(self.batch_size, device=self.device)

        # Compute Q(s', a') using the target network, but only for non-final next states
        with torch.no_grad():
            # Get Q-values from the target network for non-final next states
            next_q_values = self.target_net(non_final_next_states).max(1)[0]

            # Compute the target Q-values using the Bellman equation:
            # target = reward + gamma * max_a' Q(s', a')
            expected_q_values = reward_batch.squeeze(-1) + (1 - dones) * (self.discount_factor * next_q_values)

        # Compute the mean squared error loss between predicted Q(s, a) and target Q-values
        loss = F.mse_loss(q_values, expected_q_values)

        # Return the computed loss for backpropagation
        return loss

    def update_policy(self):
        """
        Update the policy using the calculated loss.

        Returns:
            float: Loss value after the update.
        """
        if len(self.memory) < self.batch_size:
            return
        # Generate a sample batch
        sample = self.generate_sample(self.batch_size)
        if sample is None:
            return
        non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch = sample

        # Compute loss
        loss = self.calculate_loss(non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch)
        l2_reg = torch.norm(self.policy_net.fc1.weight, p=2) + torch.norm(self.policy_net.fc2.weight, p=2)
        loss += 1e-4 * l2_reg

        # Perform gradient descent step
        # ========= put your code here ========= #
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
        # ====================================== #
    
    def update_target_networks(self):
        """
        Soft update of target network weights using Polyak averaging.
        """
        # Retrieve the state dictionaries (weights) of both networks
        # ========= put your code here ========= #
        policy_state_dict = self.policy_net.state_dict()
        target_state_dict = self.target_net.state_dict()
        # ====================================== #
        
        # Apply the soft update rule to each parameter in the target network
        # ========= put your code here ========= #
        for key in target_state_dict:
            target_state_dict[key] = (
                self.tau * policy_state_dict[key] +
                (1.0 - self.tau) * target_state_dict[key]
            )
        # ====================================== #
        
        # Load the updated weights into the target network
        # ========= put your code here ========= #
        self.target_net.load_state_dict(target_state_dict)
        # ====================================== #

    def learn(self, env, max_steps):
        """
        Train the agent on a single step.

        Args:
            env: The environment to train in.
        """

        # ===== Initialize trajectory collection variables ===== #
        # Reset environment to get initial state (tensor)
        # Track total episode return (float)
        # Flag to indicate episode termination (boolean)
        # Step counter (int)
        # ========= put your code here ========= #

        # === Episode Initialization ===
        # - Reset the environment and extract the initial observation
        # - Convert observation to tensor and set initial state
        # - Initialize reward tracker, timestep counter, and done flag
        done = False
        obs, _ = env.reset()
        total_reward = 0.0
        timestep = 0
        total_loss = 0.0
        # ====================================== #

        while not done or timestep < max_steps:
            obs_list = torch.tensor([obs['policy'][0, i] for i in range(4)], dtype=torch.float32).to(self.device)
            state = torch.tensor(obs_list, dtype=torch.float32, device=self.device).unsqueeze(0) 
            # === Action Selection ===
            # - Select action index using epsilon-greedy strategy
            # - Convert index to actual action value if necessary
            action_idx = self.select_action(obs_list)
            action = self.scale_action(action_idx).view(1, -1)

            # === Environment Interaction ===
            # - Take one step in the environment
            # - Extract next state, reward, and done flags
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_obs_list = torch.tensor([next_obs['policy'][0, i] for i in range(4)], dtype=torch.float32).to(self.device)
            
            done = terminated or truncated

            next_state = torch.tensor(next_obs_list, dtype=torch.float32, device=self.device)

            reward_tensor = torch.as_tensor(reward, dtype=torch.float32, device=self.device)

            # === Store Transition ===
            # - Ensure state/action tensor shapes are correct
            # - Store (s, a, r, s', done) in the replay buffer
            self.memory.add(state, action_idx, reward_tensor, next_state, done)

            # === Learning and Target Update ===
            # - Update Q-network if buffer has enough samples
            # - Soft update target network every fixed interval
           
            if len(self.memory) > 1000: # Perform one step of the optimization (on the policy network)
                loss = self.update_policy()
            
            # Soft update of the target network's weights
            self.update_target_networks()

            # === Bookkeeping ===
            # - Track total reward, step counters, and check termination
            total_reward += reward
            timestep += 1
            obs = next_obs

            # === End-of-Episode Handling ===
            # - Decay epsilon for exploration
            # - Plot training progress
            if done:
                self.decay_epsilon()
                return timestep, total_reward
            
    def save_w(self, path, filename):
        """
        Save weight parameters.
        """
        # ========= put your code here ========= #
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, filename)
        torch.save(self.policy_net.state_dict(), file_path)
        print(f"[INFO] Saved model weights to {file_path}")
        # ====================================== #
            
    def load_w(self, path, filename):
        """
        Load weight parameters.
        """
        # ========= put your code here ========= #
        file_path = os.path.join(path, filename)

        if os.path.exists(file_path):
            self.policy_net.load_state_dict(torch.load(file_path, map_location=self.device))
            self.target_net.load_state_dict(self.policy_net.state_dict())  # Sync target with policy
            print(f"[INFO] Loaded model weights from {file_path}")
        else:
            raise FileNotFoundError(f"[ERROR] File not found: {file_path}")
        # ====================================== #