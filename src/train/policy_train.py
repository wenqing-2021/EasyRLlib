import torch
import numpy as np
import gymnasium as gym
from abc import ABC, abstractmethod
from gymnasium.spaces import Box, Discrete
from src.config.configure import RunConfig
from src.utils.logx import EpochLogger
from src.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params
from src.utils.mpi_tools import (
    proc_id,
    num_procs,
)
from src.common.base_agent import OnPolicyAgent, OffPolicyAgent, BaseAgent
from src.common.buffer import OffPolicyBuffer, OnPolicyBuffer
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
            next_obs, rew, done, _, info = render_env.step(act)
            obs = next_obs
            if done:
                obs, _ = render_env.reset()
                render_times -= 1

        render_env.close()


class OffPolicyTrain(BaseTrainer):
    def __init__(self, configure: RunConfig = None, env: gym.Env = None) -> None:
        super().__init__(configure, env)
        self.update_every = (
            self.configure.train_config.off_policy_train_config.update_every
        )
        self.soft_update_every = (
            self.configure.train_config.off_policy_train_config.soft_update_every
        )

    def _init_env(self) -> np.ndarray:
        return super()._init_env()

    def _random_explore(self, buffer: OffPolicyBuffer) -> None:
        obs, _ = self.env.reset()
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
        local_steps_per_epoch = int(
            self.configure.train_config.total_steps
            / self.configure.train_config.epochs
            / num_procs()
        )
        obs = self._init_env()
        for steps in range(local_steps_per_epoch):
            act = agent.act(obs)
            next_obs, rew, done, _, info = self.env.step(act)
            if steps % self.update_every == 0:
                batch_data = buffer.get()
                agent.learn(batch_data)
            if steps % self.soft_update_every == 0:
                agent.soft_update()
            buffer.store(obs, act, next_obs, rew, done)
            if done:
                self.logger.store(EpRet=self._ep_ret, EpLen=self._ep_len)
                obs = self._init_env()
            else:
                obs = next_obs
                self._ep_ret += rew
                self._ep_len += 1

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
        self._set_random_seed()

        # Random Exploration
        self.logger.log("Start to colect random action data...\n", color="green")
        self._random_explore(buffer)

        # Main loop: collect experience in env and update agent with off-policy
        self.logger.log("Start to train the model!\n", color="green")
        for epoch in range(self.configure.train_config.epochs):
            self._agent_explore_learn(buffer=buffer, agent=agent)

            # store the nessarry information
            self.logger.log_tabular("Epoch", epoch)
            self.logger.log_tabular("EpRet", with_min_and_max=True)
            self.logger.log_tabular("EpLen", average_only=True)
            self.logger.dump_tabular()


class OnPolicyTrain(BaseTrainer):
    def __init__(self, configure: RunConfig = None, env: gym.Env = None) -> None:
        super().__init__(configure, env)
        self._ep_len = 0
        self._ep_ret = 0

    def _init_env(self) -> np.ndarray:
        obs, _ = self.env.reset()
        self._ep_ret, self._ep_len = 0, 0

        return obs

    def train(self, agent: OnPolicyAgent = None) -> None:
        self._log_agent_param(agent)

        # Setup buffer
        # act_dim = self._get_act_dim()
        buffer_size = int(
            self.configure.train_config.total_steps
            / self.configure.train_config.epochs
            / num_procs()
        )
        buffer = OnPolicyBuffer(
            buffer_size=buffer_size,
            batch_size=buffer_size,
            obs_shape=self.env.observation_space.shape,
            act_shape=self.env.action_space.shape,
            device=agent.device,
        )

        # Random Seed
        self._set_random_seed()

        # Main loop: collect rollout episode in env and update agent with on-policy
        self.logger.log("Start to train the model!\n", color="green")
        for epoch in range(self.configure.train_config.epochs):
            self._agent_explore_learn(buffer=buffer, agent=agent)

            # store the nessarry information
            self.logger.log_tabular("Epoch", epoch)
            self.logger.log_tabular("EpRet", with_min_and_max=True)
            self.logger.log_tabular("EpLen", average_only=True)
            self.logger.log_tabular("LossPi", average_only=True)
            self.logger.log_tabular("LossV", average_only=True)
            self.logger.dump_tabular()
            if self.configure.train_config.render:
                self._render_env(agent)

    def _agent_explore_learn(
        self, buffer: OnPolicyBuffer = None, agent: OnPolicyAgent = None
    ) -> None:
        max_steps = buffer.buffer_size
        max_ep_steps = self.configure.train_config.on_policy_train_config.max_ep_len

        obs = self._init_env()
        for steps in range(max_steps):
            act, log_pi = agent.evaluate(obs)
            state_v = agent.calc_state_value(obs)
            next_obs, rew, done, _, info = self.env.step(act)
            buffer.store(obs, act, next_obs, rew, done, log_pi, state_v)
            obs = next_obs
            self._ep_ret += rew
            self._ep_len += 1

            if done or (steps + 1) == max_ep_steps or (steps + 1) == max_steps:
                if done:
                    last_state_v = 0
                else:
                    last_state_v = agent.calc_state_value(obs)
                buffer.finish_path(
                    last_state_v=last_state_v,
                    gamma=self.configure.agent_config.gamma,
                    lam=self.configure.agent_config.lam,
                )
                self.logger.store(EpRet=self._ep_ret, EpLen=self._ep_len)
                obs = self._init_env()

        agent.learn(buffer.get())
