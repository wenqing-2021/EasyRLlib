from src.train.policy_train import BaseTrainer
from src.config.configure import RunConfig, load_config
from src.utils.mpi_tools import mpi_fork
from src import agent_lib
import torch
import argparse
import gymnasium as gym


# create the environment
def make_env(env_name: str):
    env = gym.make(env_name)
    return env


# create the agent
def make_agent(
    agent_name: str = None, env: gym.Env = None, run_config: RunConfig = None
):
    if agent_name not in agent_lib.AGENT_MAP.keys():
        raise ValueError(
            f"The defined Agent not supported. Please load agent from {list(agent_lib.AGENT_MAP.keys())} "
        )
    trainer: BaseTrainer = agent_lib.TRAINER_TYPE[agent_name](run_config, env)

    agent = agent_lib.AGENT_MAP[agent_name](
        env.observation_space, env.action_space, run_config, trainer.logger
    )

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
    args = parser.parse_args()
    run_config = load_config(args.config_path)

    mpi_fork(run_config.train_config.num_envs)  # run parallel code with mpi

    main(run_config)
