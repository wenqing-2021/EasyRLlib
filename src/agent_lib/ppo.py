from common.base_agent import OnPolicyAgent
from common.buffer import BufferData
from common.networks import MLPCategoricalActor, MLPGaussianActor, MLPCritic
from config.configure import RunConfig

import gymnasium as gym
from gymnasium import Space
from gymnasium.spaces import Box, Discrete
from typing import Tuple
import numpy as np
import torch


class PPO(OnPolicyAgent):
    def __init__(
        self,
        observation_space: gym.Space = None,
        action_space: gym.Space = None,
        configure: RunConfig = None,
    ) -> None:
        super().__init__(observation_space, action_space, configure.device)
        self.obs_dim = observation_space.shape[0]
        if isinstance(action_space, Discrete):
            self.act_dim = action_space.n
            self.policy = MLPCategoricalActor(
                self.obs_dim,
                self.act_dim,
                configure.agent_config.hidden_sizes,
                configure.agent_config.activation,
            )
        elif isinstance(action_space, Box):
            self.act_dim = action_space.shape[0]
            self.policy = MLPGaussianActor(
                self.obs_dim,
                self.act_dim,
                configure.agent_config.hidden_sizes,
                configure.agent_config.activation,
            )
        else:
            raise ValueError("ONLY Discrete or Box action space is supported")

        self.critic = MLPCritic(
            self.obs_dim,
            configure.agent_config.hidden_sizes,
            configure.agent_config.activation,
        )
        self.target_kl = configure.agent_config.target_kl

    def act(self, obs) -> Tuple[np.ndarray]:
        with torch.no_grad():
            pi = self.policy._distribution(obs)
            a = pi.sample()
            logp_a = self.policy._log_prob_from_distribution(pi, a)

        return a.cpu().numpy(), logp_a.cpu().numpy()

    def learn(self, batch_data: BufferData) -> None:
        pass

    def calc_state_value(self, obs) -> np.ndarray:
        with torch.no_grad():
            state_value = self.critic.forward(obs)
        return state_value.cpu().numpy()

    def _calc_pi_loss(self):
        pass

    def _calc_v_loss(self):
        pass
