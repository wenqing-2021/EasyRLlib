from src.train.policy_train import BaseTrainer
from src.config.configure import RunConfig, load_config, AGENT_MAP
from src.utils.mpi_tools import mpi_fork
from src.agent_lib import AgentFactory, AGENT_LIB_MAP
from src.train import TrainerFactory, TRAIN_MAP
import torch
import argparse
import gymnasium as gym
import os


def setup_agent():
    # register agent
    AgentFactory.register_agent("DQN", AGENT_LIB_MAP["DQN"])
    AgentFactory.register_agent("PPO", AGENT_LIB_MAP["PPO"])
    AgentFactory.register_agent("SAC", AGENT_LIB_MAP["SAC"])
    AgentFactory.register_agent("DDPG", AGENT_LIB_MAP["DDPG"])
    AgentFactory.register_agent("TD3", AGENT_LIB_MAP["TD3"])

    # register trainer
    TrainerFactory.register_trainer("DQN", TRAIN_MAP[AGENT_MAP["DQN"]["train"]])
    TrainerFactory.register_trainer("PPO", TRAIN_MAP[AGENT_MAP["PPO"]["train"]])
    TrainerFactory.register_trainer("SAC", TRAIN_MAP[AGENT_MAP["SAC"]["train"]])
    TrainerFactory.register_trainer("DDPG", TRAIN_MAP[AGENT_MAP["DDPG"]["train"]])
    TrainerFactory.register_trainer("TD3", TRAIN_MAP[AGENT_MAP["TD3"]["train"]])


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
    parser.add_argument(
        "-n",
        "--num_envs",
        type=int,
        default=4,
        help="Number of parallel environments",
        required=False,
    )
    args = parser.parse_args()
    debug_mode = bool(os.getenv("DEBUG_MODE", False)) or args.debug
    if debug_mode:
        torch.autograd.set_detect_anomaly(True)
        args.num_envs = 1
    mpi_fork(args.num_envs)  # run parallel code with mpi
    setup_agent()
    run_config = load_config(args.config_path)

    main(run_config)
