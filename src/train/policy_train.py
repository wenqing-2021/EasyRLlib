import torch
import numpy as np
import gymnasium as gym
from abc import ABC, abstractmethod
from gymnasium.spaces import Box, Discrete
from src.config.configure import RunConfig
from src.utils.logx import EpochLogger
from src.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params
from src.utils.mpi_tools import (
    mpi_avg,
    proc_id,
    mpi_statistics_scalar,
    num_procs,
)
from src.common.base_agent import OnPolicyAgent, OffPolicyAgent, BaseAgent
from src.common.buffer import OffPolicyBuffer, OnPolicyBuffer
from src.common.networks import count_vars


class BaseTrainer(ABC):
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

    @abstractmethod
    def train(self, agent: BaseAgent) -> None:
        """
        description: train the agent
        """
        pass

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


class OffPolicyTrain(BaseTrainer):
    def __init__(self, configure: RunConfig = None, env: gym.Env = None) -> None:
        super().__init__(configure, env)
        self.update_every = (
            self.configure.train_config.off_policy_train_config.update_every
        )
        self.soft_update_every = (
            self.configure.train_config.off_policy_train_config.soft_update_every
        )
        self.steps_per_epoch = int(
            self.configure.train_config.total_steps / self.configure.train_config.epochs
        )

    def _random_explore(self, buffer: OffPolicyBuffer, seed) -> None:
        obs, _ = self.env.reset(seed=seed)
        for _ in range(
            self.configure.train_config.off_policy_train_config.random_explor_steps
        ):
            act = self.env.action_space.sample()
            next_obs, rew, done, _, info = self.env.step(act)
            buffer.store(obs, act, next_obs, rew, done)
            if done:
                obs, _ = self.env.reset()
            else:
                obs = next_obs

    def _agent_explore_learn(
        self, buffer: OffPolicyBuffer = None, agent: OffPolicyAgent = None
    ) -> None:
        obs, _ = self.env.reset()
        ep_rew, ep_steps = 0, 0
        for steps in range(self.steps_per_epoch):
            act = agent.act(obs)
            next_obs, rew, done, _, info = self.env.step(act)
            if steps % self.update_every == 0:
                batch_data = buffer.get()
                agent.learn(batch_data)
            if steps % self.soft_update_every == 0:
                agent.soft_update()
            buffer.store(obs, act, next_obs, rew, done)
            if done:
                obs, _ = self.env.reset()
                self.logger.store(EpRet=ep_rew, EpLen=ep_steps)
                ep_rew, ep_steps = 0, 0
            else:
                obs = next_obs
                ep_rew += rew
                ep_steps += 1

    def train(self, agent) -> None:
        self._log_agent_param(agent)

        # Setup buffer
        # act_dim = self._get_act_dim()
        buffer = OffPolicyBuffer(
            buffer_size=self.configure.train_config.buffer_size,
            batch_size=self.configure.train_config.batch_size,
            obs_shape=self.env.observation_space.shape,
            act_shape=self.env.action_space.shape,
            device=agent.device,
        )

        # Random Seed
        seed = self._set_random_seed()

        # Reset the environment
        obs, _ = self.env.reset()
        local_epochs = int(self.configure.train_config.epochs / num_procs())

        # Random Exploration
        self.logger.log("Start to colect random action data...\n", color="green")
        self._random_explore(buffer, seed)

        # Main loop: collect experience in env and update agent with off-policy
        self.logger.log("Start to train the model!\n", color="green")
        for epoch in range(local_epochs):
            self._agent_explore_learn(buffer=buffer, agent=agent)

            # store the nessarry information
            self.logger.log_tabular("Epoch", epoch)
            self.logger.log_tabular("EpRet", with_min_and_max=True)
            self.logger.log_tabular("EpLen", average_only=True)
            self.logger.dump_tabular()


class OnPolicyTrain(BaseTrainer):
    def __init__(self, configure: RunConfig = None, env: gym.Env = None) -> None:
        super().__init__(configure, env)

    def train(self, agent: OnPolicyAgent = None) -> None:
        self._log_agent_param(agent)

        # Setup buffer
        # act_dim = self._get_act_dim()
        buffer = OnPolicyBuffer(
            buffer_size=self.configure.train_config.buffer_size,
            batch_size=self.configure.train_config.batch_size,
            obs_shape=self.env.observation_space.shape,
            act_shape=self.env.action_space.shape,
            device=agent.device,
        )

        # Random Seed
        seed = self._set_random_seed()

        # Reset the environment
        obs, _ = self.env.reset()
        local_epochs = int(self.configure.train_config.epochs / num_procs())
        local_steps_per_epoch = int(
            self.configure.train_config.total_steps
            / self.configure.train_config.epochs
            / num_procs()
        )

        # Main loop: collect rollout episode in env and update agent with on-policy
        self.logger.log("Start to train the model!\n", color="green")
        for epoch in range(local_epochs):
            ep_rew, ep_steps = 0, 0
            obs, ep_ret, ep_len = self.env.reset(seed=seed), 0, 0
            for steps in range(local_steps_per_epoch):
                act, log_pi = agent.act(obs)
                state_v = agent.calc_state_value(obs)
                next_obs, rew, done, _, info = self.env.step(act)
                buffer.store(obs, act, next_obs, rew, done, log_pi, state_v)
                obs = next_obs
                ep_rew += rew
                ep_steps += 1
                ep_ret += rew
                ep_len += 1

                if done or (steps + 1) == local_steps_per_epoch:
                    buffer.finish_path(last_val=0)
                    self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                    obs, ep_ret, ep_len = self.env.reset(seed=seed), 0, 0

            agent.learn(buffer.get())

            # store the nessarry information
            self.logger.log_tabular("Epoch", epoch)
            self.logger.log_tabular("EpRet", with_min_and_max=True)
            self.logger.log_tabular("EpLen", average_only=True)
            self.logger.dump_tabular()
