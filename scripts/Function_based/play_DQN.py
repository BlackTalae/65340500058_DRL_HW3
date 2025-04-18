"""Script to train RL agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from RL_Algorithm.Function_based.DQN import DQN

from tqdm import tqdm
import json

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
    num_of_action = 7
    action_range = [-20.0, 20.0]  
    learning_rate = 0.0001
    n_episodes = 50
    initial_epsilon = 0.0
    epsilon_decay = 0.0003 
    final_epsilon = 0.0
    discount = 0.01
    buffer_size = 1000
    batch_size = 32
    tau = 0.001


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

    agent = DQN(
        device=device,
        num_of_action=num_of_action,
        action_range=action_range,
        learning_rate=learning_rate,
        initial_epsilon = initial_epsilon,
        epsilon_decay = epsilon_decay,
        final_epsilon = final_epsilon,
        discount_factor = discount,
        buffer_size=buffer_size,
        batch_size=batch_size,
        tau=tau
    )

    task_name = str(args_cli.task).split('-')[0]  # Stabilize, SwingUp
    Algorithm_name = "DQN"  
    episode = 4900
    w_file = f"{Algorithm_name}_{episode}_{agent.num_of_action}_{agent.action_range[1]}_{agent.discount_factor}_{agent.lr}_{agent.epsilon_decay}_{agent.hidden_dim}_{agent.buffer_size}_{agent.batch_size}_{agent.tau}.pth"
    full_path = os.path.join(f"w/{task_name}", Algorithm_name)
    agent.load_w(full_path, w_file)

    # reset environment
    obs, _ = env.reset()
    timestep = 0

    # List สำหรับเก็บ obs
    obs_list = []

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():

            sum_step = []

            for episode in range(n_episodes):
                obs, _ = env.reset()
                done = False
                episode_obs = []  # เก็บ obs ของแต่ละ episode
                step = 0
                while not done:
                    episode_obs.append([float(obs['policy'][0, i]) for i in range(4)])
                    obs = torch.tensor([obs['policy'][0, i] for i in range(4)], dtype=torch.float32).to(agent.device)
                    # agent stepping
                    action = agent.select_action(obs)
                    scaled_action = agent.scale_action(action).view(1, -1)

                    # env stepping
                    next_obs, reward, terminated, truncated, _ = env.step(scaled_action)

                    done = terminated or truncated
                    obs = next_obs

                    step += 1

                # print("Episode:", episode+1, " Step: ",step)
                sum_step.append(step)

                # บันทึก obs ของ episode นี้
                obs_list.append(episode_obs)            

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        break

    # บันทึก obs ลง JSON
    os.makedirs("saved_obs", exist_ok=True)
    obs_file_path = os.path.join("saved_obs", f"{Algorithm_name}_{agent.num_of_action}_{agent.action_range[1]}_{agent.discount_factor}_{agent.lr}_{agent.epsilon_decay}_{agent.hidden_dim}_{agent.buffer_size}_{agent.batch_size}_{agent.tau}.json")

    # 🔧 ก่อน json.dump()
    obs_list = [obs.detach().cpu().tolist() if isinstance(obs, torch.Tensor) else obs for obs in obs_list]
    
    with open(obs_file_path, "w") as f:
        json.dump(obs_list, f, indent=4)

    print(f"Observations saved to {obs_file_path}")

    max_n = np.argmax(sum_step)

    # เลือก list ย่อยที่ต้องการ (เช่น list ที่ 2)
    sublist = obs_list[max_n]  # เปลี่ยน index ตามต้องการ

    # ดึงค่าที่ 2 ของแต่ละ list ใน list ย่อย
    y_values = [item[1] for item in sublist] #*180.0/pi

    # สร้างค่า x เป็น [1, 2, 3, ..., n]
    x_values = list(range(1, len(sublist) + 1))

    # พล็อตกราฟ
    plt.figure(figsize=(6, 4))
    plt.plot(x_values, y_values, linestyle='-', color='b', label="pole_pose")

    plt.xlabel("Step")
    plt.ylabel("Pole_pose (rad)")
    plt.title(f"{Algorithm_name}_{agent.num_of_action}_{agent.action_range[1]}_{agent.discount_factor}_{agent.lr}_{agent.epsilon_decay}_{agent.hidden_dim}_{agent.buffer_size}_{agent.batch_size}_{agent.tau}")
    plt.ylim(-0.4, 0.4)
    plt.legend()
    plt.grid(True)

    save_dir = "plots"
    os.makedirs(save_dir, exist_ok=True)  # สร้างโฟลเดอร์ถ้ายังไม่มี

    save_path = os.path.join(save_dir, f"3_{Algorithm_name}_{agent.num_of_action}_{agent.action_range[1]}_{agent.discount_factor}_{agent.lr}_{agent.epsilon_decay}_{agent.hidden_dim}_{agent.buffer_size}_{agent.batch_size}_{agent.tau}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📸 Saved plot as {save_path}")

    plt.show()


    # ==================================================================== #

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main() # type: ignore
    # close sim app
    simulation_app.close()