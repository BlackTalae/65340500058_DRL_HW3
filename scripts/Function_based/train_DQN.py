"""Script to train RL agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from RL_Algorithm.Function_based.DQN import DQN
import wandb
from tqdm import tqdm

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from datetime import datetime
import random

import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import itertools


from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
# from omni.isaac.lab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import CartPole.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

steps_done = 0

@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with stable-baselines agent."""
    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # ==================================================================== #
    # ========================= Can be modified ========================== #

    # hyperparameters
    n_episodes = 5000
    initial_epsilon = 1.0
    final_epsilon = 0.001

    # set up matplotlib
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()

    # if GPU is to be used
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    print("device: ", device)

    task_name = str(args_cli.task).split('-')[0]  # Stabilize, SwingUp
    Algorithm_name = "DQN"

    # Define hyperparameter grid
    param_grid = {
        "num_of_action":[7],
        "action_range":[20.0],        
        "learning_rate": [0.0001],
        "epsilon_decay": [0.0003],
        "discount":[0.5, 0.99],        
        "hidden_dim": [128],
        "buffer_size":[1000],
        "batch_size":[32],
        "tau":[0.001]
    }

    # Create all combinations
    grid = list(itertools.product(*param_grid.values()))
    param_names = list(param_grid.keys())

    for config_idx, values in enumerate(grid):
        config = dict(zip(param_names, values))
        print(f"\n===== Training Config {config_idx+1}/{len(grid)}: {config} =====")

        Experiment = "Discount "+str(config["discount"])
        
        # Initialize Weights and Biases (wandb) for tracking and logging metrics during training
        wandb.init( # type: ignore
            project='DRL_HW3',  # The name of the project in wandb
            name=Algorithm_name+"_"+Experiment  # The name of the current run
        )

        agent = DQN(
            device=device,
            num_of_action=config["num_of_action"],
            action_range=[-config["action_range"],config["action_range"]],
            learning_rate=config["learning_rate"],
            hidden_dim=config["hidden_dim"],
            initial_epsilon=initial_epsilon,
            epsilon_decay=config["epsilon_decay"],
            final_epsilon=final_epsilon,
            discount_factor=config["discount"],
            buffer_size=config["buffer_size"],
            batch_size=config["batch_size"],
            tau=config["tau"]
        )

        # reset environment
        obs, _ = env.reset()
        timestep = 0
        sum_step = 0
        sum_reward = 0.0

        # simulate environment
        while simulation_app.is_running():

            for episode in tqdm(range(n_episodes)): # type: ignore
                # print(agent.epsilon)
                step, reward = agent.learn(env, 1000)

                sum_step += step
                sum_reward += reward
                wandb.log({ # type: ignore
                    'num_step': step, # Steps taken in the current episode
                    'reward': reward
                })

                if episode % 100 == 0: # type: ignore
                    print(sum_step / 100.0)
                    
                    wandb.log({ # type: ignore
                        'avg_step': sum_step / 100.0,
                        'avg_reward': sum_reward / 100.0
                    })
                    sum_step = 0.0
                    sum_reward = 0.0

                    # Save Q-Learning agent
                    w_file = f"{Algorithm_name}_{episode}_{agent.num_of_action}_{agent.action_range[1]}_{agent.discount_factor}_{agent.lr}_{agent.epsilon_decay}_{agent.hidden_dim}_{agent.buffer_size}_{agent.batch_size}_{agent.tau}.pt" # type: ignore
                    full_path = os.path.join(f"w/{task_name}", Algorithm_name)
                    agent.save_w(full_path, w_file)

            print('Complete')
            if args_cli.video:
                timestep += 1
                # Exit the play loop after recording one video
                if timestep == args_cli.video_length:
                    break

            break

        # Finish the wandb run and save the logged metrics
        wandb.finish() # type: ignore 
 
    # ==================================================================== #

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main() # type: ignore
    # close sim app
    simulation_app.close()