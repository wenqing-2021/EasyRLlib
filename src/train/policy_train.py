import torch
import numpy as np
import gymnasium as gym
from abc import ABC, abstractmethod
from gymnasium.spaces import Box, Discrete
from src.config.configure import RunConfig
from src.utils.logx import EpochLogger
from src.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params
from src.utils.mpi_tools import proc_id
from src.common.base_agent import BaseAgent
from src.common.networks import count_vars


class BaseTrainer(ABC):
    def __init__(self, configure: RunConfig = None, env: gym.Env = None) -> None:
        self.configure = configure
        self.env = env
        self._ep_ret, self._ep_len = 0, 0
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

    @abstractmethod
    def train(self, agent: BaseAgent) -> None:
        """
        description: train the agent
        """
        pass

    def _init_env(self) -> np.ndarray:
        obs, _ = self.env.reset()
        self._ep_ret, self._ep_len = 0, 0

        return obs

    def _get_act_dim(self):
        act_dim = None
        if isinstance(self.env.action_space, Discrete):
            act_dim = self.env.action_space.n
        elif isinstance(self.env.action_space, Box):
            act_dim = self.env.action_space.shape[0]
        else:
            raise ValueError("The defined Action Space not supported")

        return act_dim

    def _log_agent_param(self, agent) -> None:
        # Setup agent and count vars
        self.logger.setup_pytorch_saver(agent)
        sync_params(agent)
        var_counts = count_vars(agent)
        self.logger.log("\nNumber of the model parameters: %d\n" % var_counts)

    def _set_random_seed(self) -> int:
        seed = self.configure.train_config.seed + 10000 * proc_id()
        torch.manual_seed(seed)
        np.random.seed(seed)

        return seed

    def _render_env(self, agent: BaseAgent) -> None:
        render_env = gym.make(self.configure.env_config.env_name, render_mode="human")
        obs, _ = render_env.reset()
        render_times = 10
        while render_times > 0:
            act = agent.act(obs)
            next_obs, rew, done, truncted, info = render_env.step(act)
            obs = next_obs
            if done or truncted:
                obs, _ = render_env.reset()
                render_times -= 1

        render_env.close()
