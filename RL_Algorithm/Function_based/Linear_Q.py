from __future__ import annotations
import numpy as np
import torch
from RL_Algorithm.RL_base_function import BaseAlgorithm
import matplotlib
import matplotlib.pyplot as plt

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
        self.lr = learning_rate
        self.episode_durations = []

        self.w = torch.zeros((n_observations, num_of_action), dtype=torch.float32, device=self.device)

        # set up matplotlib
        self.is_ipython = 'inline' in matplotlib.get_backend()
        if self.is_ipython:
            from IPython import display

        plt.ion()
        
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
        phi_s = obs.clone().detach().to(self.device)  # current features
        phi_next = next_obs.clone().detach().to(self.device) if not terminated else torch.zeros_like(phi_s)

        # Estimate current and target Q-values
        current_q = self.q(phi_s, a=action)
        next_q = torch.max(self.q(phi_next)).item() if not terminated else 0.0

        td_target = reward + self.discount_factor * next_q
        td_error = td_target - current_q.item()

        # Gradient update
        self.w[:, action] += self.lr * td_error * phi_s

        # Track training error
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
        if torch.rand(1).item() < self.epsilon:
            return torch.randint(0, self.num_of_action, (1,)).item()
        else:
            q_values = self.q(state_tensor)
            return torch.argmax(q_values).item()
        # ====================================== #

    def q(self, obs, a=None):
        if obs.dim() == 1:
            obs = obs.view(1, -1)  # make it [1, obs_dim]
        q_values = obs @ self.w  
        # print(q_values)

        if a is None:
            return q_values.squeeze(0)
        else:
            return q_values[0, a]

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
        obs = env.reset()
        obs_list = torch.tensor([obs[0]['policy'][0, i] for i in range(4)], dtype=torch.float32).to(self.device)
        total_reward = 0.0
        done = False
        steps = 0

        while not done or steps < max_steps:
            action = self.select_action(obs_list)
            scaled_action = self.scale_action(action).view(1, -1)
            next_obs, reward, terminated, truncated, _  = env.step(scaled_action)
            print(reward)

            next_obs_list = torch.tensor([next_obs['policy'][0, i] for i in range(4)], dtype=torch.float32).to(self.device)

            done = terminated or truncated
            self.update(obs_list, action, reward, next_obs_list, done)

            obs = next_obs
            total_reward += reward
            steps += 1


            if done:
                self.decay_epsilon()
                # print("Steps:", steps)
                # print(self.epsilon)
                self.plot_durations(steps)
                break
        
            
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

    