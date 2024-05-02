'''
Author: wenqing-2021 yuansj@hnu.edu.cn
Date: 2024-05-02 15:31:03
LastEditors: wenqing-2021 yuansj@hnu.edu.cn
LastEditTime: 2024-05-02 15:53:24
FilePath: /EasyRLlib/src/common/base_agent.py
Description: base agent for EasyRLlib
'''
import torch
from torch import nn
import numpy as np
from abc import ABC, abstractmethod
import gymnasium as gym
from gymnasium.spaces import Box, Discrete


class BaseAgent(ABC, nn.Module):
    def __init__(self, observation_space:gym.Space, action_space:gym.Space, device:str = None) -> None:
        '''
        description: 
        param {*} self
        param {gym} observation_space: gymnasium observation space
        param {gym} action_space: gymnasium action space
        param {str} device: "cpu" or "cuda"
        return {*}
        '''
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
    @abstractmethod
    def act(self, obs, act) -> np.ndarray:
        '''
        description: get the action from the actor network
        return {*}
        '''        
        pass

    @abstractmethod
    def update_net(self, *args, **kwargs):
        '''
        description: update the network
        return {*}
        '''        
        pass

    @abstractmethod
    def _calc_pi_loss(self, *args, **kwargs):
        '''
        description: calculate the policy loss
        return {*}
        '''        
        pass

    def _optimizer_update(self, optimizer:torch.optim.Optimizer, loss:torch.Tensor, clip_grad_norm:float = None):
        '''
        description: 
        param {*} self
        param {torch.optim.Optimizer} optimizer
        param {torch.Tensor} loss
        param {float} clip_grad_norm: clip the gradient norm
        return {*}
        '''
        if clip_grad_norm is not None:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(parameters=optimizer.param_groups[0]["params"], max_norm=clip_grad_norm)
            optimizer.step()
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def _soft_update(self, target_net: torch.nn.Module, current_net: torch.nn.Module, tau: float):
        '''
        description: 
        param {*} self
        param {torch.nn.module} target_net
        param {torch.nn.module} current_net
        param {float} tau
        return {*}
        '''
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

