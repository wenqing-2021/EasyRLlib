from src.common.networks import QCritic, MLPActor
from src.agent_lib.base_agent import OffPolicyAgent
from src.agent_lib.ddpg import DDPG
from src.common.buffer import BufferData
from src.config.configure import RunConfig
from src.utils.mpi_pytorch import mpi_avg_grads
from src.utils.logx import EpochLogger
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from torch import nn
import torch
import copy


class TD3(DDPG):
    def __init__(
        self,
        observation_space: gym.Space = None,
        action_space: gym.Space = None,
        configure: RunConfig = None,
        logger: EpochLogger = None,
    ) -> None:
        super().__init__(observation_space, action_space, configure, logger)
        self.target_noise = configure.agent_config.target_noise
        self.noise_clip = configure.agent_config.noise_clip
        self.policy_delay = configure.agent_config.policy_delay
        self.update_policy = 0
        # build critics
        self.critic = QCritic(
            self.obs_dim, self.act_dim, self.hidden_sizes, self.activation, 2
        ).to(self.device)
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(), lr=self.critic_lr
        )

        # create target networks
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self._set_soft_update(
            [self.actor_target, self.critic_target],
            [self.actor, self.critic],
            configure.agent_config.tau,
        )

    def _calc_critic_loss(self, batch_data: BufferData) -> torch.Tensor:
        obs = batch_data.obs
        act = batch_data.act
        next_obs = batch_data.next_obs
        with torch.no_grad():
            next_act_mu = self.actor_target(next_obs)
            next_act = self._get_act(next_act_mu)

            # Target policy smoothing
            epsilon = torch.clamp(
                torch.randn_like(next_act) * self.target_noise,
                -self.noise_clip,
                self.noise_clip,
            )
            next_act = torch.clamp(next_act + epsilon, -self.act_limit, self.act_limit)

            next_q = self.critic_target(next_obs, next_act).min(dim=-1, keepdim=True)[0]
            target_q = batch_data.rew + self.gamma * (1 - batch_data.done) * next_q

        current_q = self.critic(obs, act)
        loss_q = self.mse(current_q, target_q)

        return loss_q.mean()

    def learn(self, batch_data: BufferData) -> None:
        # update critic
        critic_loss = self._calc_critic_loss(batch_data)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        mpi_avg_grads(self.critic)
        self.critic_optimizer.step()

        if self.update_policy % self.policy_delay == 0:
            # update actor
            actor_loss = self._calc_actor_loss(batch_data)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            mpi_avg_grads(self.actor)
            self.actor_optimizer.step()

        self.update_policy += 1
