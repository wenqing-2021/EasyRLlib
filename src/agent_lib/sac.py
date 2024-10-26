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
        self.gamma = configure.agent_config.gamma

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
        self._set_soft_update(
            [self.critic_target], [self.critic], tau=configure.agent_config.tau
        )

    def act(self, obs) -> np.ndarray:
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        _, _, act = self.policy.forward(obs, with_logprob=False)

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

        _, logp_act, act = self.policy.forward(obs)
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
            _, logp_act, next_act = self.policy.forward(next_obs)
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
        self.log_alpha = torch.tensor([0.0], requires_grad=True, device=self.device)

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
        self._set_soft_update(
            [self.critic_target], [self.critic], tau=configure.agent_config.tau
        )

    def _get_all_act_prob(self, obs: torch.Tensor) -> torch.Tensor:
        pi, _, _ = self.policy.forward(obs, with_logprob=True)
        act_probs = pi.probs  # [batch, act_dim]
        log_probs = torch.log(act_probs + (act_probs == 0).float() * 1e-6)

        return act_probs, log_probs

    def _calc_actor_loss(self, batch_data: BufferData) -> torch.Tensor:
        obs = batch_data.obs
        current_alpha = self.log_alpha.exp().detach().clone().to(self.device)
        # get all action_log probs
        act_probs, log_probs = self._get_all_act_prob(obs)
        q_values = self.critic.forward(obs)  # [q_num, batch, act_dim]
        min_q = torch.min(q_values, dim=0)[0]  # [batch, act_dim]
        actor_loss = (act_probs * (current_alpha * log_probs - min_q)).sum(-1).mean()
        entropy = -(act_probs * log_probs).sum(-1).mean()

        return actor_loss, entropy

    def _calc_critic_loss(self, batch_data: BufferData) -> torch.Tensor:
        next_obs = batch_data.next_obs
        act = batch_data.act
        rew = batch_data.rew
        done = batch_data.done

        with torch.no_grad():
            act_probs, log_probs = self._get_all_act_prob(next_obs)
            next_q_values = self.critic_target.forward(next_obs).min(dim=0)[0]
            current_alpha = self.log_alpha.exp().detach().clone().to(self.device)
            target_next_q = (
                act_probs * (next_q_values - current_alpha * log_probs)
            ).sum(-1, keepdim=True)
            target_q = rew + self.gamma * (1 - done) * target_next_q
        target_qs = target_q.unsqueeze(0).repeat(self.q_num, 1, 1)
        q_values = self.critic.forward(batch_data.obs).gather(
            dim=-1, index=act.long().unsqueeze(1).unsqueeze(0).repeat(self.q_num, 1, 1)
        )
        critic_loss = self.mse(q_values, target_qs)  # [q_num, batch, 1]
        critic_loss = critic_loss.squeeze().mean(dim=-1).sum()

        return critic_loss

    def act(self, obs) -> np.ndarray:
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            _, _, act = self.policy.forward(obs=obs, with_logprob=False)

        return act.detach().cpu().numpy()

    def learn(self, batch_data: BufferData) -> None:
        loss_pi, entropy = self._calc_actor_loss(batch_data)
        loss_critic = self._calc_critic_loss(batch_data)

        alpha_loss = -(
            self.log_alpha * (self.target_entropy - entropy.detach()).detach()
        ).mean()

        # update policy
        self.policy_optimizer.zero_grad()
        loss_pi.backward()
        mpi_avg_grads(self.policy)
        self.policy_optimizer.step()

        # update critic
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        mpi_avg_grads(self.critic)
        self.critic_optimizer.step()

        # update alpha
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.logger.store(
            LossPi=loss_pi.cpu().item(),
            LossQ=loss_critic.cpu().item(),
            LossAlpha=alpha_loss.cpu().item(),
            Entropy=entropy.cpu().item(),
        )
