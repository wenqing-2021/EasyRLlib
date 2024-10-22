from src.common.networks import QNet
from src.agent_lib.base_agent import OffPolicyAgent
from src.common.buffer import BufferData
from src.config.configure import RunConfig
from src.utils.mpi_pytorch import mpi_avg_grads
from src.utils.logx import EpochLogger
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from torch import nn
import torch
import copy


class DQN(OffPolicyAgent):
    def __init__(
        self,
        observation_space: gym.Space = None,
        action_space: gym.Space = None,
        configure: RunConfig = None,
        logger: EpochLogger = None,
    ) -> None:
        super().__init__(observation_space, action_space, configure, logger)
        if not isinstance(action_space, Discrete):
            raise ValueError("ONLY Discrete action space is supported")

        self.obs_dim = observation_space.shape[0]
        self.act_dim = action_space.n
        self.hidden_sizes = configure.agent_config.hidden_sizes
        self.policy_lr = configure.agent_config.policy_lr
        self.activation = configure.agent_config.activation
        self.epsilon = configure.agent_config.epsilon
        self.gamma = configure.agent_config.gamma
        self.q_num = configure.agent_config.q_num
        self.policy = QNet(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden_sizes=self.hidden_sizes,
            activation=self.activation,
            q_num=self.q_num,
        ).to(self.device)
        self.target_q = copy.deepcopy(self.policy)
        self.policy_optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=self.policy_lr
        )
        self._set_soft_update(
            [self.target_q], [self.policy], tau=configure.agent_config.tau
        )

    def act(self, obs) -> np.ndarray:
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if np.random.rand() >= self.epsilon:
            action = self.policy.forward(obs).squeeze()
            mean_action = action.mean(dim=0, keepdim=False)
            action = mean_action.argmax(dim=-1, keepdim=False).cpu().numpy()
        else:
            action = np.random.randint(0, self.act_dim)

        return action

    def _calc_critic_loss(self, batch_data: BufferData) -> torch.Tensor:
        with torch.no_grad():
            obs = batch_data.obs
            act = batch_data.act
            next_obs = batch_data.next_obs
            next_q_list = self.target_q.forward(
                next_obs
            )  # [q_num, batch_size, act_dim]
            min_next_q = torch.min(next_q_list, dim=0)[0]
            next_q = min_next_q.max(dim=-1, keepdim=True)[0]
            target_q = batch_data.rew + self.gamma * (1 - batch_data.done) * next_q
        act_index = (
            act.long().unsqueeze(1).unsqueeze(0).expand(self.q_num, -1, -1)
        )  # [q_num, batch_size, 1]
        q = self.policy.forward(obs).gather(dim=-1, index=act_index)
        loss = 0
        for i in range(self.q_num):
            loss += self.mse(q[i], target_q).mean()

        return loss

    def learn(self, batch_data: BufferData) -> None:
        loss = self._calc_critic_loss(batch_data)
        self.logger.store(Loss=loss.cpu().item())

        self.policy_optimizer.zero_grad()
        loss.backward()
        mpi_avg_grads(self.policy)
        self.policy_optimizer.step()
