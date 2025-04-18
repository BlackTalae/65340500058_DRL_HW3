import random
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions.normal import Normal
from torch.nn.functional import mse_loss
from RL_Algorithm.RL_base_function import BaseAlgorithm
import os

class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=1e-4):
        """
        Actor network for policy approximation.

        Args:
            input_dim (int): Dimension of the state space.
            hidden_dim (int): Number of hidden units in layers.
            output_dim (int): Dimension of the action space.
            learning_rate (float, optional): Learning rate for optimization. Defaults to 1e-4.
        """
        super(Actor, self).__init__()

        # ========= put your code here ========= #

        # Define the neural network structure
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )

        self.mean = nn.Linear(hidden_dim, output_dim)
        # Set up the optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Initialize network weights
        self.init_weights()
        # ====================================== #

    def init_weights(self):
        """
        Initialize network weights using Xavier initialization for better convergence.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # Xavier initialization
                nn.init.zeros_(m.bias)  # Initialize bias to 0

    def forward(self, state):
        """
        Forward pass for action selection.

        Args:
            state (Tensor): Current state of the environment.

        Returns:
            Tensor: Selected action values.
        """
        # ========= put your code here ========= #
        x = self.net(state)
        x = torch.softmax(x, dim=-1)
        return x
        # ====================================== #

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, learning_rate=1e-4):
        """
        Critic network for Q-value approximation.

        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            hidden_dim (int): Number of hidden units in layers.
            learning_rate (float, optional): Learning rate for optimization. Defaults to 1e-4.
        """
        super(Critic, self).__init__()

        # ========= put your code here ========= #

        # Define the neural network for value prediction
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Output is scalar V(s)
        )

        # Set up the optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Initialize weights 
        self.init_weights()
        # ====================================== #

    def init_weights(self):
        """
        Initialize network weights using Kaiming initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # Kaiming initialization
                nn.init.zeros_(m.bias)  # Initialize bias to 0

    def forward(self, state, action=None):
        """
        Forward pass for Q-value estimation.

        Args:
            state (Tensor): Current state of the environment.
            action (Tensor): Action taken by the agent.

        Returns:
            Tensor: Estimated Q-value.
        """
        # ========= put your code here ========= #
        return self.net(state).squeeze(-1)  # output shape: [batch]
        # ====================================== #

class A2C_Discrete(BaseAlgorithm):
    def __init__(self, 
                device = None, 
                num_of_action: int = 2,
                action_range: list = [-2.5, 2.5],
                n_observations: int = 4,
                hidden_dim = 256,
                dropout = 0.05, 
                learning_rate: float = 0.01,
                tau: float = 0.005,
                discount_factor: float = 0.95,
                buffer_size: int = 256,
                batch_size: int = 1,
                entropy_coeff: float = 0.01,
                ):
        """
        Actor-Critic algorithm implementation.

        Args:
            device (str): Device to run the model on ('cpu' or 'cuda').
            num_of_action (int, optional): Number of possible actions. Defaults to 2.
            action_range (list, optional): Range of action values. Defaults to [-2.5, 2.5].
            n_observations (int, optional): Number of observations in state. Defaults to 4.
            hidden_dim (int, optional): Hidden layer dimension. Defaults to 256.
            learning_rate (float, optional): Learning rate. Defaults to 0.01.
            tau (float, optional): Soft update parameter. Defaults to 0.005.
            discount_factor (float, optional): Discount factor for Q-learning. Defaults to 0.95.
            batch_size (int, optional): Size of training batches. Defaults to 1.
            buffer_size (int, optional): Replay buffer size. Defaults to 256.
        """

        
        # Set device and instantiate networks
        self.device = device
        self.actor = Actor(n_observations, hidden_dim, num_of_action, learning_rate).to(device)
        self.critic = Critic(n_observations, num_of_action, hidden_dim, learning_rate).to(device)

        # Store hyperparameters
        self.discount_factor = discount_factor
        self.action_range = action_range
        self.entropy_coeff = entropy_coeff
        self.learning_rate = learning_rate
        self.hidden = hidden_dim

        # Call parent class initializer
        super(A2C_Discrete, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
        )

    def select_action(self, state):
        """
        Selects an action based on the current policy with optional exploration noise.
        
        Args:
        state (Tensor): The current state of the environment.
        noise (float, optional): The standard deviation of noise for exploration. Defaults to 0.0.

        Returns:
            Tuple[Tensor, Tensor]: 
                - scaled_action: The final action after scaling.
                - clipped_action: The action before scaling but after noise adjustment.
        """
        # ========= put your code here ========= #
        probs = self.actor(state)                       # Get action probabilities
        probs = torch.clamp(probs, min=1e-6, max=1.0)   # Prevent extremely small probabilities
        dist = torch.distributions.Categorical(probs)   # Create Categorical distribution
        action = dist.sample()                          # Sample action from distribution
        log_prob = dist.log_prob(action).sum(dim=-1)    # Compute log-probability of chosen action
        return action, log_prob
        # ====================================== #

    def calculate_loss(self, state, action, reward, next_state, done):
        """
        Computes the loss for policy optimization.

        Args:
            - states (Tensor): The batch of current states.
            - actions (Tensor): The batch of actions taken.
            - rewards (Tensor): The batch of rewards received.
            - next_states (Tensor): The batch of next states received.
            - dones (Tensor): The batch of dones received.

        Returns:
            Tensor: Computed critic & actor loss.
        """
        # ========= put your code here ========= #
        # Convert to tensors
        values = self.critic(state)             # V(s)
        next_values = self.critic(next_state)   # V(s')

        # Compute temporal difference target
        td_target = reward + self.discount_factor * next_values

        # Advantage is the difference between TD target and current value
        advantage = td_target - values

        # Actor Loss: negative log-prob * advantage
        logit = self.actor(state)
        dist = torch.distributions.Categorical(logit)
        log_probs = dist.log_prob(action)
        actor_loss = -(log_probs * advantage.detach()).mean()

        # Critic Loss: MSE of advantage
        critic_loss = (advantage**2).mean()

        return actor_loss, critic_loss
        # ====================================== #

    def update_policy(self, state, action, reward, next_state, done): 
        """
        Update the policy using the calculated loss.

        Returns:
            float: Loss value after the update.
        """
        # ========= put your code here ========= #

        # Compute critic and actor loss
        actor_loss, critic_loss = self.calculate_loss(state, action, reward, next_state, done)
        
        # Backpropagate and update critic network parameters
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # Backpropagate and update actor network parameters
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()
        # ====================================== #
        return actor_loss, critic_loss

    def learn(self, env, max_steps, num_agents):
        """
        Train the agent on a single step.

        Args:
            env: The environment in which the agent interacts.
            max_steps (int): Maximum number of steps per episode.
            num_agents (int): Number of agents in the environment.
            noise_scale (float, optional): Initial exploration noise level. Defaults to 0.1.
            noise_decay (float, optional): Factor by which noise decreases per step. Defaults to 0.99.
        """

        # ===== Initialize trajectory collection variables ===== #
        # Reset environment to get initial state (tensor)
        # Track total episode return (float)
        # Flag to indicate episode termination (boolean)
        # Step counter (int)
        # ========= put your code here ========= #

        # Reset environment
        state, _ = env.reset()
        total_reward = 0.0
        num_step = 0
        total_actorloss = 0.0
        total_criticloss = 0.0
        # ====================================== #

        for step in range(max_steps):


            # Convert observation to tensor
            state_tensor = torch.tensor([state['policy'][0, i] for i in range(4)], dtype=torch.float32).to(self.device)
            
            # Select action from actor
            action, log_prob = self.select_action(state_tensor)

            # Scale and apply action in the environment
            scaled_action = self.scale_action(action.item()).view(1, -1)
            next_state, reward, terminated, truncated, _ = env.step(scaled_action)
            
            # Process next state
            done = terminated or truncated
            next_state_tensor = torch.tensor([next_state['policy'][0, i] for i in range(4)], dtype=torch.float32).to(self.device)
           
            # Update networks using this transition                        
            actor_loss, critic_loss = self.update_policy(state_tensor, action, reward, next_state_tensor, done)

            # Update state
            state = next_state
            total_reward += reward.item()
            total_actorloss += actor_loss
            total_criticloss += critic_loss
            num_step = step

            if done:
                break

        return num_step, total_reward

    def save_w(self, path, filename):
        """
        Save model weight.
        """
        # ========= put your code here ========= #
        os.makedirs(path, exist_ok=True)
        file_path_actor = os.path.join(path, filename)

        # print(self.policy_net.state_dict())
        torch.save(self.actor.state_dict(), file_path_actor)
        print(f"[INFO] Saved model weights to {file_path_actor}")
        # ====================================== #
            
    def load_w(self, path, filename):
        """
        Load model weight.
        """
        # ========= put your code here ========= #
        file_path_actor = os.path.join(path, filename)

        if os.path.exists(file_path_actor):
            self.actor.load_state_dict(torch.load(file_path_actor, map_location=self.device))
            print(f"[INFO] Loaded model weights from {file_path_actor}")
        else:
            raise FileNotFoundError(f"[ERROR] File not found: {file_path_actor}")
        # ====================================== #