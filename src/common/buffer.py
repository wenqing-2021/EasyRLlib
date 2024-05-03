from abc import ABC, abstractmethod
import torch
import numpy as np


class BaseBuffer(ABC):
    def __init__(self, buffer_size: int = None, batch_size: int = None) -> None:
        super().__init__()
        self.buffer_size = buffer_size
        self.batch_size = batch_size
