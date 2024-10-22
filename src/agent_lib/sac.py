from copy import deepcopy
from src.agent_lib.base_agent import OffPolicyAgent
from src.common.buffer import BufferData
from src.common.networks import (
    MLPSquashedGaussianActor,
    QCritic,
    QNet,
    MLPCategoricalActor,
)
from src.config.configure import RunConfig
from src.utils.mpi_pytorch import mpi_avg_grads, mpi_avg
from src.utils.logx import EpochLogger

import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np
import torch


class SAC(OffPolicyAgent):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        configure: RunConfig = None,
        logger: EpochLogger = None,
    ) -> OffPolicyAgent:
        super().__init__(observation_space, action_space, configure, logger)

    def act(self, obs) -> np.ndarray:
        raise NotImplementedError

    def learn(self, batch_data: BufferData) -> None:
        raise NotImplementedError

    @classmethod
    def make_agent(cls, observation_space, action_space, configure, logger):
        if isinstance(action_space, Box):
            return SACC(observation_space, action_space, configure, logger)
        elif isinstance(action_space, Discrete):
            return SACD(observation_space, action_space, configure, logger)
        else:
            raise ValueError("ONLY Discrete or Box action space is supported")


class SACC(SAC):
    """
    Soft Actor-Critic with Continuous Action Space
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        configure: RunConfig = None,
        logger: EpochLogger = None,
    ) -> None:
        super().__init__(observation_space, action_space, configure, logger)
        self.obs_dim = observation_space.shape[0]
        if isinstance(action_space, Box):
            self.act_dim = action_space.shape[0]
        else:
            raise ValueError("ONLY Discrete or Box action space is supported")

        self.gamma = configure.agent_config.gamma

        # creat actor and critic
        self.policy = MLPSquashedGaussianActor(
            self.obs_dim,
            self.act_dim,
            configure.agent_config.hidden_sizes,
            configure.agent_config.activation,
        ).to(self.device)
        self.critic = QCritic(
            self.obs_dim,
            self.act_dim,
            configure.agent_config.hidden_sizes,
            configure.agent_config.activation,
        ).to(self.device)
        self.critic_target = deepcopy(self.critic)
        self.policy_optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=configure.agent_config.policy_lr
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(), lr=configure.agent_config.policy_lr
        )

        self.alpha_log = torch.tensor(
            (-1,), dtype=torch.float32, requires_grad=True, device=self.device
        )  # trainable var
        self.alpha_optim = torch.optim.Adam(
            (self.alpha_log,), lr=configure.agent_config.alpha_lr
        )
        self.target_entropy = np.log(self.act_dim)

    def act(self, obs) -> np.ndarray:
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        act, _ = self.policy.forward(obs, with_logprob=False)

        return act.detach().cpu().numpy()

    def learn(self, batch_data: BufferData) -> None:
        loss_critic = self._calc_critic_loss(batch_data)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        mpi_avg_grads(self.critic)
        self.critic_optimizer.step()

        loss_pi = self._calc_actor_loss(batch_data)
        self.policy_optimizer.zero_grad()
        loss_pi.backward()
        mpi_avg_grads(self.policy)
        self.policy_optimizer.step()

    def _calc_actor_loss(self, batch_data: BufferData) -> torch.Tensor:
        obs = batch_data.obs

        act, logp_act = self.policy.forward(obs)
        q_values = self.critic.forward(obs, act)
        min_q = torch.min(q_values, dim=-1, keepdim=True)[0]  # [batch_size, 1]
        # update alpha
        loss_alpha = (self.alpha_log * (self.target_entropy - logp_act).detach()).mean()
        self.alpha_optim.zero_grad()
        loss_alpha.backward()
        mpi_avg_grads(self.alpha_log)
        self.alpha_optim.step()

        alpha = self.alpha_log.exp().detach()
        loss_pi = (alpha * logp_act - min_q).mean()

        return loss_pi

    def _calc_critic_loss(self, batch_data: BufferData) -> torch.Tensor:
        obs = batch_data.obs  # [batch_size, obs_dim]
        act = batch_data.act
        next_obs = batch_data.next_obs
        rew = batch_data.rew  # [batch_size, 1]
        done = batch_data.done  # [batch_size, 1]

        q_values = self.critic.forward(obs, act)

        with torch.no_grad():
            # Target actions come from *current* policy
            next_act, logp_act = self.policy.forward(next_obs)
            alpha = self.alpha_log.exp()
            # Target Q-values
            next_q_values = self.critic_target.forward(
                next_obs, next_act
            )  # [batch_size, q_num]
            next_q = torch.min(next_q_values, dim=-1, keepdim=True)[0]
            target_q = rew + self.gamma * (1 - done) * (next_q - alpha * logp_act)

        # MSE loss against Bellman backup
        target_q = target_q.expand_as(q_values)  # [batch_size, q_num]
        loss_q = self.smoothL1(q_values, target_q).mean(dim=1)  # [batch_size]

        return loss_q.mean()


class SACD(SAC):
    """
    Soft Actor-Critic with Discrete Action Space
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        configure: RunConfig = None,
        logger: EpochLogger = None,
    ) -> None:
        super().__init__(observation_space, action_space, configure, logger)
        self.obs_dim = observation_space.shape[0]
        if isinstance(action_space, Discrete):
            self.act_dim = action_space.n
        else:
            raise ValueError("ONLY Discrete or Box action space is supported")
        # create policy and critic
        self.q_num = configure.agent_config.q_num
        self.policy = MLPCategoricalActor(
            self.obs_dim,
            self.act_dim,
            configure.agent_config.hidden_sizes,
            configure.agent_config.activation,
        ).to(self.device)
        self.critic = QNet(
            self.obs_dim,
            self.act_dim,
            configure.agent_config.hidden_sizes,
            configure.agent_config.activation,
            q_num=self.q_num,
        ).to(self.device)
        self.critic_target = deepcopy(self.critic)
        self.target_entropy = -self.act_dim
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()

        # create optimizers
        self.policy_optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=configure.agent_config.policy_lr
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(), lr=configure.agent_config.critic_lr
        )
        self.alpha_optimizer = torch.optim.AdamW(
            params=[self.log_alpha], lr=configure.agent_config.alpha_lr
        )

    def _calc_actor_loss(self, batch_data: BufferData) -> torch.Tensor:
        pass

    def _calc_critic_loss(self, batch_data: BufferData) -> torch.Tensor:
        pass

    def act(self, obs) -> np.ndarray:
        pass

    def learn(self, batch_data: BufferData) -> None:
        pass
