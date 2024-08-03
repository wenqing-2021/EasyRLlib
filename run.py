from train.policy_train import BaseTrainer
from config.configure import RunConfig, load_config
from utils.mpi_tools import mpi_fork
import agent_lib
import torch
import argparse
import gymnasium as gym


def update_run_config(args, run_config: RunConfig):
    # update the run_config with the arguments
    run_config.env_config.env_name = args.env_name
    run_config.exp_name = args.exp_name
    run_config.save_path = args.save_path
    run_config.train_config.seed = args.seed
    run_config.train_config.num_envs = args.num_envs


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

    agent = agent_lib.AGENT_MAP[agent_name](
        env.observation_space, env.action_space, run_config
    )
    trainer: BaseTrainer = agent_lib.TRAINER_TYPE[agent_name](run_config, env)

    return agent, trainer


# main function
def main(run_config: RunConfig):
    env = make_env(run_config.env_config.env_name)
    agent, trainer = make_agent(run_config.agent_name, env, run_config)
    trainer.train(agent)


if __name__ == "__main__":
    # load the intial configuration
    run_config = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default=run_config.env_config.env_name)
    parser.add_argument("--exp_name", type=str, default=run_config.exp_name)
    parser.add_argument("--save_path", type=str, default=run_config.save_path)
    parser.add_argument("--seed", type=int, default=run_config.train_config.seed)
    parser.add_argument(
        "--num_envs", type=int, default=run_config.train_config.num_envs
    )
    args = parser.parse_args()
    update_run_config(args, run_config)

    mpi_fork(args.num_envs)  # run parallel code with mpi

    main(run_config)
