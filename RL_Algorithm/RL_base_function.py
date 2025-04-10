import numpy as np
from collections import defaultdict, namedtuple, deque
import random
from enum import Enum
import os
import json
import torch
import torch.nn as nn

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done')) # Where is Done?

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
    
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size = 1):
        """
        Initializes the replay buffer.

        Args:
            buffer_size (int): Maximum number of experiences the buffer can hold.
            batch_size (int): Number of experiences to sample per batch.
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        """
        Adds an experience to the replay buffer.

        Args:
            state (Tensor): The current state of the environment.
            action (Tensor): The action taken at this state.
            reward (Tensor): The reward received after taking the action.
            next_state (Tensor): The next state resulting from the action.
            done (bool): Whether the episode has terminated.
        """
        is_batch = (
            isinstance(done, torch.Tensor) and done.ndim >= 1 and done.shape[0] > 1
        )

        if is_batch:
            for s, a, r, ns, d in zip(state, action, reward, next_state, done):
                self.memory.append(Transition(s, a, ns, r, d))
        else:
            self.memory.append(Transition(state, action, next_state, reward, done))

    def sample(self):
        """
        Samples a batch of experiences from the replay buffer.

        Returns:
            - state_batch: Batch of states.
            - action_batch: Batch of actions.
            - reward_batch: Batch of rewards.
            - next_state_batch: Batch of next states.
            - done_batch: Batch of terminal state flags.
        """
        transitions = random.sample(self.memory, self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.stack(batch.state).to(device)
        action_batch = torch.stack(batch.action).to(device)
        next_state_batch = torch.stack(batch.next_state).to(device)        
        reward_batch = torch.stack(batch.reward).to(device)
        done_batch = torch.tensor(batch.done, dtype=torch.bool).to(device)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        """
        Returns the current size of the replay buffer.

        Returns:
            int: The number of stored experiences.
        """
        return len(self.memory)


class BaseAlgorithm():
    """
    Base class for reinforcement learning algorithms.

    Attributes:
        num_of_action (int): Number of discrete actions available.
        action_range (list): Scale for continuous action mapping.
        discretize_state_scale (list): Scale factors for discretizing states.
        lr (float): Learning rate for updates.
        epsilon (float): Initial epsilon value for epsilon-greedy policy.
        epsilon_decay (float): Rate at which epsilon decays.
        final_epsilon (float): Minimum epsilon value allowed.
        discount_factor (float): Discount factor for future rewards.
        q_values (dict): Q-values for state-action pairs.
        n_values (dict): Count of state-action visits (for Monte Carlo method).
        training_error (list): Stores training errors for analysis.
    """

    def __init__(
        self,
        num_of_action: int = 11,
        action_range: list = [-12.0, 12.0],
        learning_rate: float = 1e-3,
        initial_epsilon: float = 1.0,
        epsilon_decay: float = 0.999,
        final_epsilon: float = 0.001,
        discount_factor: float = 0.95,
        buffer_size: int = 1000,
        batch_size: int = 32,
        n_observation: int = 4
    ):
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.num_of_action = num_of_action
        self.action_range = action_range  # [action_min, action_max]

        self.q_values = defaultdict(lambda: np.zeros(self.num_of_action))
        self.n_values = defaultdict(lambda: np.zeros(self.num_of_action))
        self.training_error = []

        self.w = torch.zeros((n_observation, num_of_action), dtype=torch.float32, device=device)
        self.memory = ReplayBuffer(buffer_size, batch_size)

    # def q(self, obs, a=None):
    #     """Returns the linearly-estimated Q-value for a given state and action."""
    #     # ========= put your code here ========= #
    #     if obs.dim() == 1:
    #         obs = obs.view(1, -1)
    #     # w_tensor = torch.as_tensor(self.w, dtype=torch.float32, device=obs.device)

    #     # print("obs.shape:", obs.shape)
    #     # print("self.w.shape:", self.w.shape)
    #     # print("action.shape:", a.shape)

    #     q_values = obs @ self.w  # [1, n_observation] @ [n_observation, num_action] → [1, num_action]

    #     if a is None:
    #         return q_values.squeeze(0) # q_value of every action 
    #     else:
    #         if isinstance(a, int):
    #             return q_values[:, a].squeeze(0) if q_values.shape[0] == 1 else q_values
    #         else:
    #             if not isinstance(a, torch.Tensor):
    #                 a = torch.tensor(a, dtype=torch.long, device=obs.device)
    #             else:
    #                 a = a.to(dtype=torch.long, device=obs.device).clone().detach()
    #             return q_values.gather(1, a.unsqueeze(1)).squeeze(1)
    
    #     # # ====================================== #
        
    
    def scale_action(self, action):
        """
        Maps a discrete action in range [0, n] to a continuous value in [action_min, action_max].

        Args:
            action (int): Discrete action in range [0, n].
            n (int): Number of discrete actions (inclusive range from 0 to n).
        
        Returns:
            torch.Tensor: Scaled action tensor.
        """
        # ========= put your code here ========= #
        # Unpack the minimum and maximum values of the action range
        action_min, action_max = self.action_range

        # Scale the discrete action index (0 to num_of_action-1) to a continuous value within [action_min, action_max]
        scaled = action_min + (action / (self.num_of_action - 1)) * (action_max - action_min)

        # Check if the scaled value is already a torch.Tensor
        if isinstance(scaled, torch.Tensor):
            # If yes, detach it from any computation graph and convert to float32
            return scaled.clone().detach().to(dtype=torch.float32)
        else:
            # Otherwise, convert it into a torch.Tensor of type float32
            return torch.tensor(scaled, dtype=torch.float32)
        # ====================================== #
    
    def decay_epsilon(self):
        """
        Decay epsilon value to reduce exploration over time.
        """
        # ========= put your code here ========= #
        # Decay the exploration rate (epsilon) by multiplying with epsilon_decay,
        # but ensure it doesn't go below the minimum value (final_epsilon)
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)
        # ====================================== #

    def save_w(self, path, filename):
        """
        Save weight parameters.
        """
        # ========= put your code here ========= #
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, filename)
        np.save(file_path, self.w)
        # ====================================== #
            
    def load_w(self, path, filename):
        """
        Load weight parameters.
        """
        # ========= put your code here ========= #
        file_path = os.path.join(path, filename)
        if os.path.exists(file_path):
            self.w = np.load(file_path)
        else:
            raise FileNotFoundError(f"Weight file not found: {file_path}")
        # ====================================== #


