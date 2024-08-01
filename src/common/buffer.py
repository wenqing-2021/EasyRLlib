from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union
import torch
import numpy as np
import gymnasium as gym


@dataclass
class BufferData:
    obs: Union[np.ndarray, torch.Tensor] = None
    act: Union[np.ndarray, torch.Tensor] = None
    next_obs: Union[np.ndarray, torch.Tensor] = None
    rew: Union[np.ndarray, torch.Tensor] = None
    done: Union[np.ndarray, torch.Tensor] = None
    device: torch.device = None

    def convert_to_tensor(self):
        for k, v in self.__dict__.items():
            if v is not None and isinstance(v, np.ndarray):
                self.__dict__[k] = torch.as_tensor(
                    v, dtype=torch.float32, device=self.device
                )

    def convert_to_array(self):
        for k, v in self.__dict__.items():
            if v is not None and isinstance(v, torch.Tensor):
                self.__dict__[k] = v.cpu().numpy()


class BaseBuffer(ABC):
    """
    description: abstract class for buffer
    return {*}
    """

    def __init__(
        self,
        buffer_size: int = None,
        batch_size: int = None,
        device: torch.device = None,
    ) -> None:
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
        self.data = BufferData(device=device)

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

    def _combined_shape(self, length, shape=None):
        if shape is None:
            return (length,)
        return (length, shape) if np.isscalar(shape) else (length, *shape)

    def _initialize(self, obs_shape=None, act_shape=None):
        self.data.obs = np.zeros(
            self._combined_shape(self.buffer_size, obs_shape), dtype=np.float32
        )
        self.data.act = np.zeros(
            self._combined_shape(self.buffer_size, act_shape), dtype=np.float32
        )
        self.data.next_obs = np.zeros(
            self._combined_shape(self.buffer_size, obs_shape), dtype=np.float32
        )
        self.data.rew = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.data.done = np.zeros((self.buffer_size, 1), dtype=np.float32)


class OffPolicyBuffer(BaseBuffer):
    def __init__(
        self,
        buffer_size: int = None,
        batch_size: int = None,
        obs_shape=None,
        act_shape=None,
        device: torch.device = None,
    ) -> None:
        super().__init__(buffer_size, batch_size, device)
        self.obs_shape = obs_shape
        self.act_shape = act_shape
        self.count = 0
        self._initialize(obs_shape, act_shape)

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
        batch_data = BufferData(device=self.data.device)
        if self.count < self.batch_size:
            for k, v in self.data.__dict__.items():
                if v is not None and isinstance(v, np.ndarray):
                    batch_data.__dict__[k] = v[: self.count]
        else:
            idx = np.random.choice(self.count, self.batch_size, replace=False)
            for k, v in self.data.__dict__.items():
                if v is not None and isinstance(v, np.ndarray):
                    batch_data.__dict__[k] = v[idx]

        return batch_data


class OnPolicyBuffer(BaseBuffer):
    def __init__(
        self,
        buffer_size: int = None,
        batch_size: int = None,
        obs_shape=None,
        act_shape=None,
        device: torch.device = None,
    ) -> None:
        super().__init__(buffer_size, batch_size, device)
        self.obs_shape = obs_shape
        self.act_shape = act_shape
        self.count = 0
        self._initialize(obs_shape, act_shape)

    def _initialize(self, obs_shape=None, act_shape=None):
        super()._initialize(obs_shape, act_shape)
        log_pi = np.zeros((self.buffer_size, 1), dtype=np.float32)
        state_value = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.data.__setattr__("log_pi", log_pi)
        self.data.__setattr__("state_value", state_value)

    def clear(self):
        self._initialize(self.obs_shape, self.act_shape)

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        next_obs: np.ndarray,
        rew: np.ndarray,
        done: np.ndarray,
        log_p: np.ndarray,
        state_value: np.ndarray,
    ):
        self.data.obs[self.count] = obs
        self.data.act[self.count] = act
        self.data.next_obs[self.count] = next_obs
        self.data.rew[self.count] = rew
        self.data.done[self.count] = done
        self.data.log_pi[self.count] = log_p
        self.data.state_value[self.count] = state_value
        self.count += 1
        if self.count >= self.buffer_size:
            self.count = 0

    def get(self, last_state_value: np.ndarray) -> BufferData:
        self._finish_path(last_state_value)
        batch_data = BufferData(device=self.data.device)

    def _finish_path(self, last_state_value: np.ndarray) -> None:
        pass
