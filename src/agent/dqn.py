from common.networks import QNet
from common.base_agent import BaseAgent
from common.buffer import BufferData
from config.configure import RunConfig
from utils.mpi_pytorch import mpi_avg_grads
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from torch import nn
import torch
import copy


class DQN(BaseAgent):
    def __init__(
        self,
        observation_space: gym.Space = None,
        action_space: Discrete = None,
        configure: RunConfig = None,
    ) -> None:
        super().__init__(observation_space, action_space)
        if not isinstance(action_space, Discrete):
            raise ValueError("ONLY Discrete action space is supported")

        self.obs_dim = observation_space.shape[0]
        self.act_dim = action_space.n
        self.hidden_sizes = configure.agent_config.hidden_sizes
        self.q_lr = configure.agent_config.q_lr
        self.activation = configure.agent_config.activation
        self.epsilon = configure.agent_config.epsilon
        self.gamma = configure.agent_config.gamma
        self.policy = QNet(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden_sizes=self.hidden_sizes,
            activation=self.activation,
        ).to(self.device)
        self.target_q = copy.deepcopy(self.policy)
        self.policy_optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=self.q_lr
        )
        self.set_soft_update(
            [self.target_q], [self.policy], tau=configure.agent_config.tau
        )

    def act(self, obs) -> np.ndarray:
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if np.random.rand() >= self.epsilon:
            action = self.policy.forward(obs)
            action = action.argmax(dim=-1, keepdim=False).cpu().numpy()
        else:
            action = np.random.randint(0, self.act_dim)

        return action

    def learn(self, batch_data: BufferData) -> None:
        batch_data.convert_to_tensor()
        with torch.no_grad():
            obs = batch_data.obs
            act = batch_data.act
            next_obs = batch_data.next_obs
            next_q = self.target_q.forward(next_obs).max(dim=-1, keepdim=True)[0]
            target_q = batch_data.rew + self.gamma * (1 - batch_data.done) * next_q

        q = self.policy.forward(obs).gather(dim=-1, index=act.long().unsqueeze(1))
        loss = nn.functional.mse_loss(q, target_q)

        self.policy_optimizer.zero_grad()
        loss.backward()
        mpi_avg_grads(self.policy)
        self.policy_optimizer.step()
