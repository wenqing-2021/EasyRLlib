from gymnasium import Space
from numpy import ndarray
from common.base_agent import BaseAgent
from common.networks import MLPCategoricalActor, MLPGaussianActor, MLPCritic
import gymnasium as gym
from config.configure import RunConfig
from gymnasium.spaces import Box, Discrete


class PPO(BaseAgent):
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

    def act(self, obs, act) -> ndarray:
        pass
