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
        if isinstance(action_space, Discrete):
            self.act_dim = action_space.n
        elif isinstance(action_space, Box):
            self.act_dim = action_space.shape[0]
        else:
            raise ValueError("ONLY Discrete or Box action space is supported")

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

    def act(self, obs) -> np.ndarray:
        pass

    def learn(self, batch_data: BufferData) -> None:
        pass
