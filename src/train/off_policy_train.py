from src.train.policy_train import BaseTrainer
from src.config.configure import RunConfig
from src.utils.mpi_tools import num_procs
from src.common.base_agent import OffPolicyAgent
from src.common.buffer import OffPolicyBuffer

import numpy as np
import gymnasium as gym


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
            next_obs, rew, done, truncted, info = self.env.step(act)
            if steps % self.update_every == 0:
                batch_data = buffer.get()
                agent.learn(batch_data)
            if steps % self.soft_update_every == 0:
                agent.soft_update()
            buffer.store(obs, act, next_obs, rew, done)
            if done or truncted or steps == local_steps_per_epoch - 1:
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
