import torch
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from config.configure import RunConfig
from utils.logx import EpochLogger
from utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from utils.mpi_tools import (
    mpi_avg,
    proc_id,
    mpi_statistics_scalar,
    num_procs,
)
from common.base_agent import BaseAgent
from common.buffer import OffPolicyBuffer
from common.networks import count_vars


class BaseTrainer:
    def __init__(self, configure: RunConfig = None, env: gym.Env = None) -> None:
        self.configure = configure
        self.env = env
        # Special function to avoid certain slowdowns from PyTorch + MPI combo.
        setup_pytorch_for_mpi()

        # Create logger
        logger_kwargs = EpochLogger.setup_logger_kwargs(
            exp_name=self.configure.exp_name,
            seed=self.configure.train_config.seed,
            data_dir=self.configure.save_path,
        )
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())

    def _get_act_dim(self):
        act_dim = None
        if isinstance(self.env.action_space, Discrete):
            act_dim = self.env.action_space.n
        elif isinstance(self.env.action_space, Box):
            act_dim = self.env.action_space.shape[0]
        else:
            raise ValueError("The defined Action Space not supported")

        return act_dim


class OffPolicyTrain(BaseTrainer):
    def __init__(self, configure: RunConfig = None, env: gym.Env = None) -> None:
        super().__init__(configure, env)
        self.update_every = self.configure.train_config.update_every

    def _random_explore(self, buffer: OffPolicyBuffer, seed) -> None:
        obs, _ = self.env.reset(seed=seed)
        for _ in range(self.configure.train_config.random_explor_steps):
            act = self.env.action_space.sample()
            next_obs, rew, done, _, info = self.env.step(act)
            buffer.store(obs, act, next_obs, rew, done)
            if done:
                obs, _ = self.env.reset()
            else:
                obs = next_obs

    def train(self, agent: BaseAgent = None) -> None:
        # Setup agent and count vars
        self.logger.setup_pytorch_saver(agent)
        sync_params(agent)
        var_counts = count_vars(agent)
        self.logger.log("\nNumber of the model parameters: %d\n" % var_counts)

        # Setup buffer
        act_dim = self._get_act_dim()
        buffer = OffPolicyBuffer(
            buffer_size=self.configure.train_config.buffer_size,
            batch_size=self.configure.train_config.batch_size,
            obs_dim=self.env.observation_space.shape[0],
            act_dim=act_dim,
        )

        # Random Seed
        seed = self.configure.train_config.seed + 10000 * proc_id()
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Reset the environment
        obs, _ = self.env.reset()
        steps_per_epoch = int(
            self.configure.train_config.total_steps / self.configure.train_config.epochs
        )
        local_epochs = int(self.configure.train_config.epochs / num_procs())

        # Random Exploration
        self.logger.log("Start to colect random action data.", color="green")
        self._random_explore(buffer, seed)

        # Main loop: collect experience in env and update agent with off-policy
        self.logger.log("Start to train the model!", color="green")
        return


class OnPolicyTrain:
    pass
