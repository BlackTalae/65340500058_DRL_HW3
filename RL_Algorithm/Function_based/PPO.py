import random
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions.normal import Normal
from torch.nn.functional import mse_loss
from RL_Algorithm.RL_base_function import BaseAlgorithm

import matplotlib
import matplotlib.pyplot as plt

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
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, output_dim)
        self.log_std = nn.Parameter(torch.zeros(output_dim))  # Learnable log std
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
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
        if torch.isnan(state).any() or torch.isinf(state).any():
            print("❗ NaN or Inf in input state:", state)

        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        if torch.isnan(x).any():
            print("❗ NaN detected in hidden layer")
        mu = self.mu(x)
        std = self.log_std.exp().expand_as(mu)
        std = torch.clamp(std, min=1e-6, max=1.0)  # ✅ ป้องกันค่าพุ่งหรือเป็น 0

        return Normal(mu, std)  # Return distribution object
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
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
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

    def forward(self, state):
        """
        Forward pass for Q-value estimation.

        Args:
            state (Tensor): Current state of the environment.
            action (Tensor): Action taken by the agent.

        Returns:
            Tensor: Estimated Q-value.
        """
        # ========= put your code here ========= #
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        value = self.v(x)
        return value
        # ====================================== #

class Actor_Critic_PPO(BaseAlgorithm):
    def __init__(self, 
                device = None, 
                num_of_action: int = 1,
                n_observations: int = 4,
                hidden_dim = 256,
                dropout = 0.05, 
                learning_rate: float = 0.01,
                tau: float = 0.005,
                discount_factor: float = 0.95,
                eps_clip: float = 0.2,
                entropy_coeff: float = 0.01,
                n_epoch: int = 4
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
        """
        # Feel free to add or modify any of the initialized variables above.
        # ========= put your code here ========= #
        self.device = device
        self.actor = Actor(n_observations, hidden_dim, num_of_action, learning_rate).to(device)
        self.actor_target = Actor(n_observations, hidden_dim, num_of_action, learning_rate).to(device)
        self.critic = Critic(n_observations, num_of_action, hidden_dim, learning_rate).to(device)
        self.critic_target = Critic(n_observations, num_of_action, hidden_dim, learning_rate).to(device)

        self.tau = tau
        self.eps_clip = eps_clip
        self.discount_factor = discount_factor
        self.trajectory = []
        self.episode_durations = []

        self.update_target_networks(tau=1)  # initialize target networks
        self.entropy_coeff = entropy_coeff # or pass as argument
        self.n_epoch = n_epoch


        # Experiment with different values and configurations to see how they affect the training process.
        # Remember to document any changes you make and analyze their impact on the agent's performance.


        # ====================================== #

        super(Actor_Critic_PPO, self).__init__(
            num_of_action=num_of_action,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
        )

        # set up matplotlib
        self.is_ipython = 'inline' in matplotlib.get_backend()
        if self.is_ipython:
            from IPython import display

        plt.ion()

    def select_action(self, state, noise=0.0):
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
        dist = self.actor(state)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.detach(), log_prob.detach(), dist
        # ====================================== #
    
    def generate_sample(self):
        """
        Generates a batch sample from memory for training.

        Returns:
            Tuple: A tuple containing:
                - state_batch (Tensor): The batch of current states.
                - action_batch (Tensor): The batch of actions taken.
                - reward_batch (Tensor): The batch of rewards received.
                - next_state_batch (Tensor): The batch of next states received.
                - done_batch (Tensor): The batch of dones received.
        """
        # Ensure there are enough samples in memory before proceeding
        
        # Sample a batch from memory
        # ========= put your code here ========= #
        if len(self.trajectory) < 2:
            return None

        data = list(zip(*self.trajectory))

        states = torch.stack(data[0]).to(self.device)
        actions = torch.stack(data[1]).to(self.device)
        old_log_probs = torch.stack(data[2]).to(self.device)
        rewards = torch.stack(data[3]).unsqueeze(1).to(self.device)
        next_states = torch.stack(data[4]).to(self.device)
        dones = torch.stack(data[5]).unsqueeze(1).to(self.device)

        return states, actions, old_log_probs, rewards, next_states, dones
        # ====================================== #

    def calculate_loss(self, states, actions, old_log_probs, rewards, dones, next_states, discount_factor, eps_clip):
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
        def compute_returns(rewards, dones, gamma=0.99):
            """
            Vectorized Monte Carlo return computation with episode ends (dones).
            """

            rewards = rewards.view(-1)
            dones = dones.view(-1)

            returns = torch.zeros_like(rewards)
            G = 0.0
            for t in reversed(range(len(rewards))):
                G = rewards[t] + discount_factor * G * (1.0 - dones[t])
                returns[t] = G
            return returns
        # Update Critic

        # Gradient clipping for critic

        # Update Actor

        # Gradient clipping for actor

        # --- Compute returns (with no grad) ---
        with torch.no_grad():
            returns = compute_returns(rewards, dones, discount_factor).detach()  # returns is already a tensor

        values = self.critic(states).view(-1)  # [T]
        advantages = returns - values.detach() # [T]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dist = self.actor(states)
        new_log_probs = dist.log_prob(actions).sum(dim=-1)  # [T]
        entropy = dist.entropy().sum(dim=-1)                # [T]

        ratios = torch.exp(new_log_probs - old_log_probs)

        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
        actor_loss = -torch.min(surr1, surr2 - self.entropy_coeff * entropy).mean()
        critic_loss = mse_loss(values, returns)

        return actor_loss, critic_loss, entropy.mean()


    def update_policy(self, n_epochs, batch_size=256):
        """
        Update the policy using the calculated loss.

        Returns:
            float: Loss value after the update.
        """
        # ========= put your code here ========= #
        sample = self.generate_sample()
        if sample is None:
            return

        states, actions, old_log_probs, rewards, next_states, dones = sample
        total_size = states.shape[0]

        for epoch in range(n_epochs):
            indices = torch.randperm(total_size)
            for i in range(0, total_size, batch_size):
                idx = indices[i:i+batch_size]
                s = states[idx]
                a = actions[idx]
                lp = old_log_probs[idx]
                r = rewards[idx]
                ns = next_states[idx]
                d = dones[idx]

                actor_loss, critic_loss, entropy = self.calculate_loss(s, a, lp, r, ns, d, self.discount_factor, self.eps_clip)

                # --- Backprop for actor ---
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                self.actor.optimizer.step()

                # --- Backprop for critic ---
                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.critic.optimizer.step()

        self.trajectory.clear()
        return actor_loss.item(), critic_loss.item()
        # ====================================== #


    def update_target_networks(self, tau=None):
        """
        Perform soft update of target networks using Polyak averaging.

        Args:
            tau (float, optional): Update rate. Defaults to self.tau.
        """
        # ========= put your code here ========= #
        if tau is None:
            tau = self.tau

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        # ====================================== #

    def learn(self, env, max_steps, num_agents, noise_scale=0.1, noise_decay=0.99):
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
        state = env.reset()
        # if isinstance(state, tuple):  # for Gymnasium
        #     state = state[0]
        state = torch.tensor([state[0]['policy'][0, i] for i in range(4)], dtype=torch.float32).to(self.device)
        
        total_reward = 0.0
        done = False
        step = 0
        self.trajectory = []
        # ====================================== #
        
        while step < max_steps:
            # Predict action from the policy network
            # ========= put your code here ========= #
            action, log_prob, _ = self.select_action(state, noise=noise_scale)
            action = action.view(1, -1)
            # ====================================== #

            # Execute action in the environment and observe next state and reward
            # ========= put your code here ========= #
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state_tensor = torch.tensor([next_state['policy'][0, i] for i in range(4)], dtype=torch.float32).to(self.device)
            reward_tensor = torch.tensor([reward], dtype=torch.float32).to(self.device)
            done_tensor = torch.tensor([done], dtype=torch.float32).to(self.device)
            # ====================================== #

            # Store the transition in memory
            # ========= put your code here ========= #
            # Parallel Agents Training
            if num_agents > 1:
                raise NotImplementedError("Multi-agent PPO not implemented in this setup")
            # Single Agent Training
            else:
                self.trajectory.append((
                    torch.tensor(state, dtype=torch.float32).to(self.device) if not isinstance(state, torch.Tensor) else state,
                    torch.tensor(action, dtype=torch.float32).to(self.device) if not isinstance(action, torch.Tensor) else action,
                    log_prob.detach(),
                    reward_tensor,
                    next_state_tensor,  # ✅ เปลี่ยนจาก next_state → next_state_tensor
                    done_tensor
                ))
            # ====================================== #

            # Update state

            # Reset if needed
            if done:
                # Perform one step of the optimization (on the policy network)
                self.update_policy(self.n_epoch)

                # Update target networks
                self.update_target_networks()
                # print("Step: ",step)
                self.plot_durations(step)

                break
            else:
                state = next_state_tensor
            total_reward += reward
            step += 1
            # Decay the noise to gradually shift from exploration to exploitation
            noise_scale *= noise_decay  # decay exploration noise

    # Consider modifying this function to visualize other aspects of the training process.
    # ================================================================================== #
    def plot_durations(self, timestep=None, show_result=False):
        if timestep is not None:
            self.episode_durations.append(timestep)

        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if self.is_ipython:
            if not show_result:
                display.display(plt.gcf()) # type: ignore
                display.clear_output(wait=True) # type: ignore
            else:
                display.display(plt.gcf()) # type: ignore
    # ================================================================================== #

