from src.train.policy_train import BaseTrainer
from src.config.configure import RunConfig, load_config
from src.utils.mpi_tools import mpi_fork
from src.agent_lib import AgentFactory
from src.train import TrainerFactory
import torch
import argparse
import gymnasium as gym
import os


# create the environment
def make_env(env_name: str):
    env = gym.make(env_name)
    return env


# create the agent
def make_agent(
    agent_name: str = None, env: gym.Env = None, run_config: RunConfig = None
):
    trainer: BaseTrainer = TrainerFactory.make_trainer(agent_name, run_config, env)

    agent = AgentFactory.make_agent(agent_name, env, run_config, trainer.logger)

    return agent, trainer


# main function
def main(run_config: RunConfig):
    env = make_env(run_config.env_config.env_name)
    agent, trainer = make_agent(run_config.agent_config.agent_name, env, run_config)
    trainer.train(agent)


if __name__ == "__main__":
    # load the intial configuration
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        default="/home/EasyRLlib/experiment/dqn_cartpole/run_config.yaml",
        help="Path to the config file",
        required=False,
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        default=False,
        help="Run the code in debug mode",
        required=False,
    )
    args = parser.parse_args()
    run_config = load_config(args.config_path)
    debug_mode = bool(os.getenv("DEBUG_MODE", False))
    if debug_mode:
        torch.autograd.set_detect_anomaly(True)
        run_config.train_config.num_envs = 1
    mpi_fork(run_config.train_config.num_envs)  # run parallel code with mpi

    main(run_config)
