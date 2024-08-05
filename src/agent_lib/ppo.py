from common.base_agent import OnPolicyAgent
from common.buffer import BufferData
from common.networks import MLPCategoricalActor, MLPGaussianActor, MLPCritic
from config.configure import RunConfig
from utils.mpi_pytorch import mpi_avg_grads

import gymnasium as gym
from gymnasium import Space
from gymnasium.spaces import Box, Discrete
from typing import Tuple
import numpy as np
import torch
from torch import nn


class PPO(OnPolicyAgent):
    def __init__(
        self,
        observation_space: gym.Space = None,
        action_space: gym.Space = None,
        configure: RunConfig = None,
    ) -> None:
        super().__init__(observation_space, action_space, configure.device)
        self.obs_dim = observation_space.shape[0]
        if isinstance(action_space, Discrete):
            self.act_dim = action_space.n
            self.policy = MLPCategoricalActor(
                self.obs_dim,
                self.act_dim,
                configure.agent_config.hidden_sizes,
                configure.agent_config.activation,
            )
        elif isinstance(action_space, Box):
            self.act_dim = action_space.shape[0]
            self.policy = MLPGaussianActor(
                self.obs_dim,
                self.act_dim,
                configure.agent_config.hidden_sizes,
                configure.agent_config.activation,
            )
        else:
            raise ValueError("ONLY Discrete or Box action space is supported")
        self.policy_optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=configure.agent_config.policy_lr
        )
        self.critic = MLPCritic(
            self.obs_dim,
            configure.agent_config.hidden_sizes,
            configure.agent_config.activation,
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(), lr=configure.agent_config.value_lr
        )
        self.target_kl = configure.agent_config.target_kl
        self.clip_ratio = configure.agent_config.clip_ratio
        self.update_times = configure.train_config.on_policy_train_config.update_times

    def act(self, obs) -> Tuple[np.ndarray]:
        with torch.no_grad():
            pi = self.policy._distribution(obs)
            a = pi.sample()
            logp_a = self.policy._log_prob_from_distribution(pi, a)

        return a.cpu().numpy(), logp_a.cpu().numpy()

    def learn(self, batch_data: BufferData) -> None:
        batch_data.convert_to_tensor()

        for i in range(self.update_times):
            loss_pi = self._calc_pi_loss(batch_data)
            loss_v = self._calc_v_loss(batch_data)

            if self.logger.get_stats("KL") > 1.2 * self.target_kl:
                self.logger.log("Early stopping at step %d due to reaching max kl." % i)
                break

            self.policy_optimizer.zero_grad()
            loss_pi.backward()
            mpi_avg_grads(self.policy)
            self.policy_optimizer.step()

            self.critic_optimizer.zero_grad()
            loss_v.backward()
            mpi_avg_grads(self.critic)
            self.critic_optimizer.step()

        self.logger.store(Stopiter=i)

    def calc_state_value(self, obs) -> np.ndarray:
        with torch.no_grad():
            state_value = self.critic.forward(obs)
        return state_value.cpu().numpy()

    def _calc_pi_loss(self, batch_data: BufferData) -> torch.Tensor:
        obs = batch_data.obs
        act = batch_data.act
        advantage: torch.Tensor = batch_data.gae_adv
        log_pi_old: torch.Tensor = batch_data.log_pi

        pi = self.policy._distribution(obs)
        log_pi = self.policy._log_prob_from_distribution(pi, act)
        ratio = torch.exp(log_pi - log_pi_old)
        clip_adv = (
            torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantage
        )
        loss_pi = -torch.min(ratio * advantage, clip_adv).mean()

        approx_kl = (log_pi_old - log_pi).mean().cpu().item()
        entropy = pi.entropy().mean().cpu().item()
        clip_frac = (
            (ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio))
            .mean()
            .cpu()
            .item()
        )

        # log info
        self.logger.store(
            LossPi=loss_pi.cpu().item(),
            KL=approx_kl,
            Entropy=entropy,
            ClipFrac=clip_frac,
        )

        return loss_pi

    def _calc_v_loss(self, batch_data: BufferData) -> torch.Tensor:
        obs = batch_data.obs
        ret = batch_data.ret
        state_value = self.critic.forward(obs)
        loss_v = nn.functional.mse_loss(state_value, ret).mean()

        # log info
        self.logger.store(LossV=loss_v.cpu().item())

        return loss_v
