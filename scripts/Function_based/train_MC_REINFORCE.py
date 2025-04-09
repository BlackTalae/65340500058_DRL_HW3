"""Script to train RL agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from RL_Algorithm.Function_based.MC_REINFORCE import MC_REINFORCE

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

HYDRA_FULL_ERROR=1

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

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

# CUDA determinism
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
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
    num_of_action = 13
    action_range = [-20.0, 20.0]
    learning_rate = 0.0001
    hidden_dim = 512
    n_episodes = 5000
    initial_epsilon = 1.0
    epsilon_decay = 0.999
    final_epsilon = 0.1
    discount = 0.99
    buffer_size = 10000
    batch_size = 128
    target_update_freq = 1000

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
        "learning_rate": [1e-3],
        "hidden_dim": [128],
        "discount":[0.9],
        "dropout":[0.0]
    }

    # Create all combinations
    grid = list(itertools.product(*param_grid.values()))
    param_names = list(param_grid.keys())

    for config_idx, values in enumerate(grid):
        config = dict(zip(param_names, values))
        # print(f"\n===== Training Config {config_idx+1}/{len(grid)}: {config} =====")

        agent = MC_REINFORCE(
            device=device,
            learning_rate=config["learning_rate"],
            hidden_dim=config["hidden_dim"],
            discount_factor=config["discount"],
            dropout=config["dropout"]
        )

        # reset environment
        obs, _ = env.reset()
        timestep = 0
        # simulate environment
        while simulation_app.is_running():
            # run everything in inference mode
            # with torch.inference_mode():

            for episode in tqdm(range(n_episodes)): # type: ignore
                # print(agent.epsilon)
                agent.learn(env)

            # if episode % 100 == 0: # type: ignore
            #     print(agent.epsilon)

            #     # Save Q-Learning agent
            #     w_file = f"{Algorithm_name}_{episode}_{num_of_action}_{action_range[1]}.json" # type: ignore
            #     full_path = os.path.join(f"w/{task_name}", Algorithm_name)
            #     agent.save_w(full_path, w_file)
            print("Average Return:", sum(agent.episode_durations)/n_episodes)
            # print('Complete')
            agent.plot_durations(show_result=True)
            plt.ioff()
            # plt.show()
            # plt.savefig(f"{agent.batch_size}_{agent.learning_rate}_{agent.num_of_action}_{agent.hidden_dim}_{agent.epsilon_decay}_{agent.discount_factor}_{agent.buffer_size}.png")  # บันทึกไฟล์ภาพ
            plt.close()                # ปิด plot เพื่อไม่ให้แสดง
            if args_cli.video:
                timestep += 1
                # Exit the play loop after recording one video
                if timestep == args_cli.video_length:
                    break

            break
    # ==================================================================== #

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main() # type: ignore
    # close sim app
    simulation_app.close()