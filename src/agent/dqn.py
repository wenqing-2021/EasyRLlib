from common.networks import QNet
from common.base_agent import BaseAgent
from config.configure import RunConfig, DQNConfig
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete


class DQN(BaseAgent):
    def __init__(
        self,
        observation_space: gym.Space = None,
        action_space: Discrete = None,
        configure: RunConfig = None,
    ) -> None:
        super().__init__(observation_space, action_space)
        self.obs_dim = observation_space.shape[0]
        self.act_dim = action_space.n
        self.hidden_sizes = configure.agent_config.hidden_sizes
        self.q_lr = configure.agent_config.q_lr
        self.activation = configure.agent_config.activation
        self.policy = QNet(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden_sizes=self.hidden_sizes,
            activation=self.activation,
        )

    def act(self, obs) -> np.ndarray:
        pass

    def learn(self) -> None:
        pass
