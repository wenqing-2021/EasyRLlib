from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
import numpy as np


@dataclass
class BufferData:
    obs: np.ndarray = None
    act: np.ndarray = None
    next_obs: np.ndarray = None
    rew: np.ndarray = None
    done: np.ndarray = None


class BaseBuffer(ABC):
    """
    description: abstract class for buffer
    return {*}
    """

    def __init__(self, buffer_size: int = None, batch_size: int = None) -> None:
        """
        description: init the base buffer
        param {*} self
        param {int} buffer_size: the size of the buffer
        param {int} batch_size: the size of the batch used for training
        return {*}
        """
        super().__init__()
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.data = BufferData()

    @abstractmethod
    def store(self, *args, **kwargs):
        """
        description: store the data into the buffer
        return {*}
        """
        pass

    @abstractmethod
    def get(self, *args, **kwargs):
        """
        description: get the data from the buffer
        return {*}
        """
        pass

    def _initialize(self, obs_dim: int = None, act_dim: int = None):
        self.data.obs = np.zeros((self.buffer_size, obs_dim), dtype=np.float32)
        self.data.act = np.zeros((self.buffer_size, act_dim), dtype=np.float32)
        self.data.next_obs = np.zeros((self.buffer_size, obs_dim), dtype=np.float32)
        self.data.rew = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.data.done = np.zeros((self.buffer_size, 1), dtype=np.float32)


class OffPolicyBuffer(BaseBuffer):
    def __init__(
        self,
        buffer_size: int = None,
        batch_size: int = None,
        obs_dim: int = None,
        act_dim: int = None,
    ) -> None:
        super().__init__(buffer_size, batch_size)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.count = 0
        self._initialize(obs_dim, act_dim)

    def store(self, obs, act, next_obs, rew, done):
        self.data.obs[self.count] = obs
        self.data.act[self.count] = act
        self.data.next_obs[self.count] = next_obs
        self.data.rew[self.count] = rew
        self.data.done[self.count] = done
        self.count += 1
        if self.count == self.buffer_size:
            self.count = 0

    def get(self) -> BufferData:
        batch_data = BufferData()
        if self.count < self.batch_size:
            for k, v in self.data.__dict__.items():
                if v is not None:
                    batch_data.__dict__[k] = v[: self.count]
        else:
            idx = np.random.choice(self.count, self.batch_size, replace=False)
            for k, v in self.data.__dict__.items():
                if v is not None:
                    batch_data.__dict__[k] = v[idx]

        return batch_data
