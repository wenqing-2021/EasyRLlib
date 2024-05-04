from policy_train import OffPolicyTrain
from config.configure import RunConfig, load_config
from utils.mpi_tools import mpi_fork
import agent as agent
import torch
import argparse
import gymnasium as gym

AGENT_MAP = {"DQN": agent.DQN}


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
    try:
        agent = AGENT_MAP[agent_name](
            env.observation_space, env.action_space, run_config
        )
    except KeyError:
        raise ValueError(
            f"The defined Agent not supported. Please load agent from {list(AGENT_MAP.keys())} "
        )

    return agent


# main function
def main(run_config: RunConfig):
    env = make_env(run_config.env_config.env_name)
    agent = make_agent(run_config.agent_name, env, run_config)
    trainer = OffPolicyTrain(run_config, env)
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
