from src.common.networks import QCritic, MLPActor
from src.agent_lib.base_agent import OffPolicyAgent
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


class DDPG(OffPolicyAgent):
    def __init__(
        self,
        observation_space: gym.Space = None,
        action_space: gym.Space = None,
        configure: RunConfig = None,
        logger: EpochLogger = None,
    ) -> None:
        super().__init__(observation_space, action_space, configure, logger)
        if not isinstance(action_space, Box):
            raise ValueError("ONLY Box action space is supported in DDPG")

        self.obs_dim = observation_space.shape[0]
        self.act_dim = action_space.shape[0]
        self.act_limit = [action_space.low[0], action_space.high[0]]
        self.hidden_sizes = configure.agent_config.hidden_sizes
        self.policy_lr = configure.agent_config.policy_lr
        self.critic_lr = configure.agent_config.critic_lr
        self.activation = configure.agent_config.activation
        self.noise_std = configure.agent_config.noise_std
        self.act_norm_range = [-1.0, 1.0]
        self.gamma = configure.agent_config.gamma
        # build actor
        self.actor = MLPActor(
            self.obs_dim, self.act_dim, self.hidden_sizes, self.activation
        ).to(self.device)
        # build critics
        self.critic = QCritic(
            self.obs_dim, self.act_dim, self.hidden_sizes, self.activation, 1
        ).to(self.device)
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(), lr=self.policy_lr
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(), lr=self.critic_lr
        )

        # create target networks
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self._set_soft_update(
            [self.actor_target, self.critic_target],
            [self.actor, self.critic],
            configure.agent_config.tau,
        )

    def _get_act(self, act_mu, noise_std=0.0) -> torch.Tensor:
        def scale_act(act_norm, norm_range: list, act_limit: list):
            """
            scale the action from norm_range to the range of the action space, default norm_range is [-1, 1]
            """
            ratio = (act_norm - norm_range[0]) / (norm_range[1] - norm_range[0])
            scale_act = ratio * (act_limit[1] - act_limit[0]) + act_limit[0]

            return scale_act

        if noise_std > 0:
            act_std = noise_std * torch.ones_like(act_mu)
            act_dist = torch.distributions.normal.Normal(act_mu, act_std)
            act_norm = act_dist.sample().clip(
                self.act_norm_range[0], self.act_norm_range[1]
            )
            act = scale_act(act_norm, self.act_norm_range, self.act_limit)
        else:
            act = scale_act(act_mu, self.act_norm_range, self.act_limit)

        return act

    def act(self, obs) -> np.ndarray:
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        act_mu = self.actor(obs)
        act = self._get_act(act_mu, self.noise_std)
        act_arr = act.cpu().detach().numpy()
        return act_arr

    def learn(self, batch_data: BufferData) -> None:
        # update critic
        critic_loss = self._calc_critic_loss(batch_data)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        mpi_avg_grads(self.critic)
        self.critic_optimizer.step()

        # update actor
        actor_loss = self._calc_actor_loss(batch_data)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        mpi_avg_grads(self.actor)
        self.actor_optimizer.step()

    def _calc_actor_loss(self, batch_data: BufferData) -> torch.Tensor:
        obs = batch_data.obs
        act_mu = self.actor(obs)
        # act = self._get_act(act_mu)
        loss_pi = -self.critic(obs, act_mu).mean()

        return loss_pi

    def _calc_critic_loss(self, batch_data: BufferData) -> torch.Tensor:
        obs = batch_data.obs
        act = batch_data.act
        next_obs = batch_data.next_obs
        with torch.no_grad():
            next_act_mu = self.actor_target(next_obs)
            # next_act = self._get_act(next_act_mu)
            next_q = self.critic_target(next_obs, next_act_mu)
            target_q = batch_data.rew + self.gamma * (1 - batch_data.done) * next_q

        current_q = self.critic(obs, act)
        loss_q = self.mse(current_q, target_q)

        return loss_q.mean()
