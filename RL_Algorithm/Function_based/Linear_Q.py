from __future__ import annotations
import numpy as np
import torch
from RL_Algorithm.RL_base_function import BaseAlgorithm
import matplotlib
import matplotlib.pyplot as plt
import os
import json

class Linear_QN(BaseAlgorithm):
    def __init__(
            self,
            device: None,
            num_of_action: int = 2,
            n_observations: int = 4,
            action_range: list = [-2.5, 2.5],
            learning_rate: float = 0.01,
            initial_epsilon: float = 1.0,
            epsilon_decay: float = 1e-3,
            final_epsilon: float = 0.001,
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

        super().__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
        )

        self.device = device
        self.episode_durations = []

        # Initialize the weight matrix for linear Q-value approximation.
        # Shape: [state_dim, num_actions]
        self.w = torch.zeros((n_observations, num_of_action), dtype=torch.float32, device=self.device)
        
    def update(
        self,
        obs,
        action,
        reward,
        next_obs,
        terminated
    ):
        """
        Updates the weight vector using the Temporal Difference (TD) error 
        in Q-learning with linear function approximation.

        Args:
            obs (dict): The current state observation, containing feature representations.
            action (int): The action taken in the current state.
            reward (float): The reward received for taking the action.
            next_obs (dict): The next state observation.
            next_action (int): The action taken in the next state (used in SARSA).
            terminated (bool): Whether the episode has ended.

        """
        # ========= put your code here ========= #
        """
        Q-learning with linear function approximation.
        """
        # Convert observations to tensor and move to appropriate device
        phi_s = obs.clone().detach().to(self.device)  # current features
        phi_next = next_obs.clone().detach().to(self.device) if not terminated else torch.zeros_like(phi_s)

        # Estimate current and target Q-values
        current_q = self.q(phi_s, a=action)
        next_q = torch.max(self.q(phi_next)).item() if not terminated else 0.0

        # Calculate TD target and TD error
        td_target = reward + self.discount_factor * next_q
        td_error = td_target - current_q.item()

        # Apply gradient update to the weight vector for the taken action
        self.w[:, action] += self.lr * td_error * phi_s

        # Log TD error for analysis
        self.training_error.append(td_error)

        # ====================================== #

    def select_action(self, state):
        """
        Select an action based on an epsilon-greedy policy.
        
        Args:
            state (Tensor): The current state of the environment.
        
        Returns:
            Tensor: The selected action.
        """
        # ========= put your code here ========= #
        state_tensor = state.to(self.device)

        # Explore: with probability epsilon, select a random action
        if torch.rand(1).item() < self.epsilon:
            return torch.randint(0, self.num_of_action, (1,)).item()
        else:
            # Exploit: choose the action with the highest Q-value
            q_values = self.q(state_tensor)
            return torch.argmax(q_values).item()
        # ====================================== #

    def q(self, obs, a=None):
        # Ensure obs has batch dimension
        if obs.dim() == 1:
            obs = obs.view(1, -1)  # make it [1, obs_dim]

        # Compute linear Q-values: Q(s) = obs @ w
        q_values = obs @ self.w  

        if a is None:
            return q_values.squeeze(0) # Return Q-values for all actions
        else:
            return q_values[0, a] # Return Q-value for specific action

    def learn(self, env, max_steps):
        """
        Train the agent on a single step.

        Args:
            env: The environment in which the agent interacts.
            max_steps (int): Maximum number of steps per episode.
        """

        # ===== Initialize trajectory collection variables ===== #
        # Reset environment to get initial state (tensor)
        # Track total episode return (float)
        # Flag to indicate episode termination (boolean)
        # Step counter (int)
        # ========= put your code here ========= #

        # Reset environment and initialize counters
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        steps = 0

        # Run until episode ends or max_steps is reached
        while not done or steps < max_steps:

            # Extract state vector (shape: [4]) from observation dictionary
            obs_list = torch.tensor([obs['policy'][0, i] for i in range(4)], dtype=torch.float32).to(self.device)
            
            # Choose action based on current policy
            action = self.select_action(obs_list)

            # Scale the discrete action to match environment's action space
            scaled_action = self.scale_action(action).view(1, -1)

            # Take an action in the environment
            next_obs, reward, terminated, truncated, _  = env.step(scaled_action)

            # Extract next state vector
            next_obs_list = torch.tensor([next_obs['policy'][0, i] for i in range(4)], dtype=torch.float32).to(self.device)

            # Determine whether the episode is over
            done = terminated or truncated

            # Update weight with new experience
            self.update(obs_list, action, reward, next_obs_list, done)

            # Accumulate reward and move to next state
            obs = next_obs
            total_reward += reward
            steps += 1

            # If episode ends, decay epsilon and return stats
            if done:
                self.decay_epsilon()
                return steps, total_reward

    def save_w(self, path, filename):
        """
        Save weight parameters.
        """
        # ========= put your code here ========= #
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, filename)

        if isinstance(self.w, torch.Tensor):
            weights_list = self.w.detach().cpu().numpy().tolist()
        elif isinstance(self.w, np.ndarray):
            weights_list = self.w.tolist()
        else:
            weights_list = self.w  # already list?

        # ถ้า self.w เป็น torch.Tensor บน GPU → ต้อง .cpu() ด้วย
        weights_list = self.w.detach().cpu().tolist()

        with open(file_path, 'w') as f:
            json.dump(weights_list, f)

        print(f"[INFO] Saved weights to {file_path}")
        # ====================================== #
            
    def load_w(self, path, filename):
        """
        Load weight parameters.
        """
        # ========= put your code here ========= #
        file_path = os.path.join(path, filename)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                weights_list = json.load(f)

            # Convert to tensor on the correct device
            self.w = torch.tensor(weights_list, dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")

            print(f"[INFO] Loaded weights from {file_path}")
        else:
            raise FileNotFoundError(f"[ERROR] Weight file not found: {file_path}")
        # ====================================== #