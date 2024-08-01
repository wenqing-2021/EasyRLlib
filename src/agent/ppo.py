from gymnasium import Space
from common.base_agent import BaseAgent
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
        elif isinstance(action_space, Box):
            self.act_dim = action_space.shape[0]
        else:
            raise ValueError("ONLY Discrete or Box action space is supported")
