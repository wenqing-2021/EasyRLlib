from src.agent_lib.base_agent import OnPolicyAgent
from src.common.buffer import BufferData
from src.common.networks import MLPCategoricalActor, MLPGaussianActor, MLPCritic
from src.config.configure import RunConfig
from src.utils.mpi_pytorch import mpi_avg_grads, mpi_avg
from src.utils.logx import EpochLogger

import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from typing import Tuple
import numpy as np
import torch
from torch import nn


class A2C(OnPolicyAgent):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        configure: RunConfig = None,
        logger: EpochLogger = None,
    ) -> None:
        super().__init__(observation_space, action_space, configure, logger)
        self.obs_dim = observation_space.shape[0]
        self.update_times = configure.train_config.on_policy_train_config.update_times
        if isinstance(action_space, Discrete):
            self.act_dim = action_space.n
            self.policy = MLPCategoricalActor(
                self.obs_dim,
                self.act_dim,
                configure.agent_config.hidden_sizes,
                configure.agent_config.activation,
            ).to(self.device)
        elif isinstance(action_space, Box):
            self.act_dim = action_space.shape[0]
            self.policy = MLPGaussianActor(
                self.obs_dim,
                self.act_dim,
                configure.agent_config.hidden_sizes,
                configure.agent_config.activation,
            ).to(self.device)
        else:
            raise ValueError("ONLY Discrete or Box action space is supported")
        self.policy_optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=configure.agent_config.policy_lr
        )
        self.critic = MLPCritic(
            self.obs_dim,
            self.act_dim,
            configure.agent_config.hidden_sizes,
            configure.agent_config.activation,
        ).to(self.device)
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(), lr=configure.agent_config.critic_lr
        )

    def act(self, obs) -> np.ndarray:
        """
        get the action from the policy network
        """
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            pi = self.policy._distribution(obs)
            a = pi.sample()

        return a.cpu().numpy()

    def calc_state_value(self, obs) -> np.ndarray:
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            state_value = self.critic.forward(obs)
        return state_value.cpu().numpy()

    def evaluate(self, obs) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            pi = self.policy._distribution(obs)
            act = pi.sample()
            logp_a = self.policy._log_prob_from_distribution(pi, act)

        return act.cpu().detach().numpy(), logp_a.detach().cpu().numpy()

    def learn(self, batch_data: BufferData) -> None:
        loss_pi_info_old = self._calc_actor_loss(batch_data)
        loss_v_info_old = self._calc_critic_loss(batch_data)
        loss_pi_info = self._calc_actor_loss(batch_data)
        self.policy_optimizer.zero_grad()
        loss_pi_info["LossPi"].backward()
        mpi_avg_grads(self.policy)
        self.policy_optimizer.step()
        for i in range(self.update_times):

            loss_v_info = self._calc_critic_loss(batch_data)
            self.critic_optimizer.zero_grad()
            loss_v_info["LossV"].backward()
            mpi_avg_grads(self.critic)
            self.critic_optimizer.step()

        self.logger.store(
            LossPi=loss_pi_info_old["LossPi"].cpu().item(),
            LossV=loss_v_info_old["LossV"].cpu().item(),
        )

    def _calc_actor_loss(self, batch_data: BufferData) -> torch.Tensor:
        obs = batch_data.obs
        act = batch_data.act
        advantage: torch.Tensor = batch_data.gae_adv
        _, log_pi, _ = self.policy.forward(obs, act)
        loss_pi = -(log_pi * advantage).mean()

        # log info
        loss_pi_info = {"LossPi": loss_pi}

        return loss_pi_info

    def _calc_critic_loss(self, batch_data: BufferData) -> torch.Tensor:
        obs = batch_data.obs
        ret = batch_data.discount_ret
        state_value = self.critic.forward(obs).reshape(ret.shape)
        loss_v = self.mse(state_value, ret).mean()

        # log info
        loss_v_info = {"LossV": loss_v}

        return loss_v_info
