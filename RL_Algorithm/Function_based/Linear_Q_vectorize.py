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
        self.total_env_episodes = 0

        # เปลี่ยนจาก numpy → tensor
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
        # Ensure all inputs are batched
        obs = obs.to(self.device)  # [B, obs_dim]
        next_obs = next_obs.to(self.device)
        action = action.to(self.device)  # [B]
        reward = reward.to(self.device)  # [B]
        terminated = terminated.to(self.device)  # [B]

        # Set next_obs to 0 where terminated
        phi_next = next_obs * (~terminated).unsqueeze(1)  # [B, obs_dim]

        # Q(s, a) for current obs
        current_q = self.q(obs, a=action)  # [B]

        # Max Q(s', a') for next state
        next_q = self.q(phi_next).max(dim=1)[0]  # [B]

        td_target = reward + self.discount_factor * next_q
        td_error = td_target - current_q

        # Weight update: for each action a, accumulate td_error * phi_s
        for a in range(self.num_of_action):
            mask = (action == a)  # [B]
            if mask.any():
                delta = (td_error[mask].unsqueeze(1) * obs[mask]).sum(dim=0)  # [obs_dim]
                self.w[:, a] += self.lr * delta

        self.training_error.extend(td_error.tolist())

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
        batch_size = state_tensor.shape[0]
        if torch.rand(1).item() < self.epsilon:
            return torch.randint(0, self.num_of_action, (batch_size,), device=self.device)
        else:
            q_values = self.q(state_tensor)
            return torch.argmax(q_values, dim=1)
        # ====================================== #

    def q(self, obs, a=None):
        """Returns the linearly-estimated Q-value for a given state and action."""
        # ========= put your code here ========= #
        if obs.dim() == 1:
            obs = obs.view(1, -1)
        # w_tensor = torch.as_tensor(self.w, dtype=torch.float32, device=obs.device)

        # print("obs.shape:", obs.shape)
        # print("self.w.shape:", self.w.shape)
        # print("action.shape:", a.shape)

        q_values = obs @ self.w  # [1, n_observation] @ [n_observation, num_action] → [1, num_action]

        if a is None:
            return q_values.squeeze(0) # q_value of every action 
        else:
            if isinstance(a, int):
                return q_values[:, a].squeeze(0) if q_values.shape[0] == 1 else q_values
            else:
                if not isinstance(a, torch.Tensor):
                    a = torch.tensor(a, dtype=torch.long, device=obs.device)
                else:
                    a = a.to(dtype=torch.long, device=obs.device).clone().detach()
                return q_values.gather(1, a.unsqueeze(1)).squeeze(1)
    
        # # ====================================== #
    
    def learn(self, env, max_episodes):
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

        obs, _ = env.reset()
        obs_tensor = obs['policy'].to(self.device)
        episode_rewards = torch.zeros(obs_tensor.shape[0], device=self.device)
        episode_lengths = torch.zeros(obs_tensor.shape[0], dtype=torch.int32)

        while True:  # รันไปเรื่อย ๆ จนกว่าจะครบจำนวน episode ที่ต้องการ

            actions = self.select_action(obs_tensor)
            scaled_actions = self.scale_action(actions).view(-1, 1)
            next_obs, rewards, terms, truncs, infos = env.step(scaled_actions)

            next_obs_tensor = next_obs['policy'].to(self.device)
            done = terms | truncs  # [B]

            self.update(obs_tensor, actions, rewards, next_obs_tensor, done)

            obs_tensor = next_obs_tensor
            episode_rewards += rewards
            episode_lengths += 1  

            for i in range(obs_tensor.shape[0]):
                if done[i]:
                    # print(f"[EP {self.total_env_episodes}] Epsilon: {self.epsilon:.4f}")
                    obs_reset, _ = env.reset()
                    obs_tensor[i] = obs_reset['policy'][i].to(self.device)

                    self.total_env_episodes += 1
                    self.decay_epsilon()  # decay epsilon ทุกครั้งที่ reset env
                    self.plot_durations(episode_lengths[i].item())
                    episode_rewards[i] = 0
                    episode_lengths[i] = 0

                    if self.total_env_episodes >= max_episodes:
                        return


        # return total_reward, steps
        # ====================================== #
        
            
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

    