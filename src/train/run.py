from policy_train import OffPolicyTrain
from src.config.configure import RunConfig, load_config
from src.utils.mpi_tools import mpi_fork
import torch
import argparse
import gymnasium as gym

if __name__ == "__main__":
    run_config = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default=run_config.env_config.env_name)
    parser.add_argument("--exp_name", type=str, default=run_config.exp_name)
    parser.add_argument("--save_path", type=str, default=run_config.save_path)
    parser.add_argument("--seed", type=int, default=run_config.train_config.seed)
    parser.add_argument("--cpu", type=int, default=run_config.train_config.num_envs)
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi
