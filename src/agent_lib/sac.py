from copy import deepcopy
from src.common.base_agent import OffPolicyAgent
from src.common.buffer import BufferData
from src.common.networks import MLPSquashedGaussianActor, QCritic
from src.config.configure import RunConfig
from src.utils.mpi_pytorch import mpi_avg_grads, mpi_avg
from src.utils.logx import EpochLogger

import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from typing import Tuple
import numpy as np
import torch
from torch import nn


class SAC(OffPolicyAgent):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        configure: RunConfig = None,
        logger: EpochLogger = None,
    ) -> None:
        super().__init__(observation_space, action_space, configure.device, logger)

        self.obs_dim = observation_space.shape[0]
        if isinstance(action_space, Box):
            self.act_dim = action_space.shape[0]
        else:
            raise ValueError("ONLY Discrete or Box action space is supported")

        self.gamma = configure.agent_config.gamma

        # creat actor and critic
        self.policy = MLPSquashedGaussianActor(
            self.obs_dim,
            self.act_dim,
            configure.agent_config.hidden_sizes,
            configure.agent_config.activation,
        ).to(self.device)
        self.critic = QCritic(
            self.obs_dim,
            self.act_dim,
            configure.agent_config.hidden_sizes,
            configure.agent_config.activation,
        ).to(self.device)
        self.critic_target = deepcopy(self.critic)
        self.policy_optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=configure.agent_config.policy_lr
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(), lr=configure.agent_config.policy_lr
        )

        self.alpha_log = torch.tensor(
            (-1,), dtype=torch.float32, requires_grad=True, device=self.device
        )  # trainable var
        self.alpha_optim = torch.optim.Adam(
            (self.alpha_log,), lr=configure.agent_config.alpha_lr
        )
        self.target_entropy = np.log(self.act_dim)

    def act(self, obs) -> np.ndarray:
        act, _ = self.policy.forward(obs, with_logprob=False)

        return act.detach().cpu().numpy()

    def learn(self, batch_data: BufferData) -> None:
        pass

    def _calc_actor_loss(self, batch_data: BufferData) -> torch.Tensor:
        pass

    def _calc_critic_loss(self, batch_data: BufferData) -> torch.Tensor:
        obs = batch_data.obs
        act = batch_data.act
        next_obs = batch_data.next_obs
        rew = batch_data.rew
        done = batch_data.done

        q_values = self.critic.forward(obs, act)

        with torch.no_grad():
            # Target actions come from *current* policy
            next_act, logp_n_act = self.policy.forward(next_obs)
            alpha = self.alpha_log.exp()
            # Target Q-values
            next_q_values = self.critic_target.forward(next_obs, next_act)
            next_q = torch.min(next_q_values)
            target_q = rew + self.gamma * (1 - done) * (next_q - alpha * logp_n_act)

        # MSE loss against Bellman backup
        loss_q = self.smoothL1(q_values, target_q).mean(dim=1)

        # Useful info for logging

        return loss_q
