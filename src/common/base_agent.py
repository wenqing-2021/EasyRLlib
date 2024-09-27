"""
Author: wenqing-2021 yuansj@hnu.edu.cn
Date: 2024-05-02 15:31:03
LastEditors: wenqing-2021 yuansj@hnu.edu.cn
LastEditTime: 2024-05-03 15:34:57
FilePath: /EasyRLlib/src/common/base_agent.py
Description: base agent for EasyRLlib
"""

import torch
from torch import nn
import numpy as np
from abc import ABC, abstractmethod
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from src.common.buffer import BufferData
from src.utils.logx import EpochLogger
from typing import List, Union, Tuple


class BaseAgent(ABC, nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: str = None,
        logger: EpochLogger = None,
    ) -> None:
        """
        description:
        param {*} self
        param {gym} observation_space: gymnasium observation space
        param {gym} action_space: gymnasium action space
        param {str} device: "cpu" or "cuda"
        return {*}
        """
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.tau: float = None
        self.target_net_list: List[nn.Module] = []
        self.current_net_list: List[nn.Module] = []
        self.logger = logger
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    @abstractmethod
    def act(self, obs, act) -> np.ndarray:
        """
        description: get the action from the actor network
        return {*}
        """
        pass

    @abstractmethod
    def learn(self, batch_data: BufferData) -> None:
        """
        description: update the network using the batch data
        return {*}
        """
        pass

    def set_soft_update(
        self,
        target_net_list: List[nn.Module],
        current_net_list: List[nn.Module],
        tau: float = 0.005,
    ):
        self.target_net_list = target_net_list
        self.current_net_list = current_net_list
        self.tau = tau

    def soft_update(self):
        if self.tau is None:
            raise ValueError("The tau is not set!")
        if len(self.target_net_list) != len(self.current_net_list):
            raise ValueError("The target and current network list should be the same!")
        if len(self.target_net_list) < 1:
            raise ValueError("The target network list is empty!")
        for target_net, current_net in zip(self.target_net_list, self.current_net_list):
            for tar, cur in zip(target_net.parameters(), current_net.parameters()):
                tar.data.copy_(cur.data * self.tau + tar.data * (1.0 - self.tau))


class OffPolicyAgent(BaseAgent):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: str = None,
        logger: EpochLogger = None,
    ) -> None:
        super().__init__(observation_space, action_space, device, logger)


class OnPolicyAgent(BaseAgent):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: str = None,
        logger: EpochLogger = None,
    ) -> None:
        super().__init__(observation_space, action_space, device, logger)

    @abstractmethod
    def calc_state_value(self, obs) -> np.ndarray:
        """
        description: calculate the state value
        return {*}
        """
        pass

    @abstractmethod
    def evaluate(self, obs, act) -> Tuple[np.ndarray, np.ndarray]:
        """
        description: evaluate the action
        return {*}
        """
        pass
