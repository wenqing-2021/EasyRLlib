import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from abc import ABC, abstractmethod

LOG_STD_MIN = -20
LOG_STD_MAX = 2


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module: nn.Module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class QNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def forward(self, obs):
        q = self.q_net(obs)
        return torch.squeeze(q, -1)


class Actor(ABC, nn.Module):
    def __init__(self, obs_dim, act_dim) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.net = None

    @abstractmethod
    def _distribution(self, obs):
        pass

    @abstractmethod
    def _log_prob_from_distribution(self, pi, act):
        pass

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__(obs_dim, act_dim)
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(
        self, pi: Categorical, act: torch.Tensor
    ) -> torch.Tensor:
        return pi.log_prob(act)


class MLPGaussianActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__(obs_dim, act_dim)
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(
        self, pi: Normal, act: torch.Tensor
    ) -> torch.Tensor:
        return pi.log_prob(act).sum(
            axis=-1
        )  # Last axis sum needed for Torch Normal distribution


class MLPSquashedGaussianActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation) -> None:
        super().__init__(obs_dim, act_dim)
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_net = mlp([hidden_sizes[-1], act_dim], activation=nn.Identity)
        self.log_std_net = mlp([hidden_sizes[-1], act_dim], activation=nn.Identity)

    def _distribution(self, obs):
        net_out = self.net(obs)
        mu = self.mu_net(net_out)
        log_std = self.log_std_net(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(
        self, pi: Normal, act: torch.Tensor
    ) -> torch.Tensor:
        logp_pi = pi.log_prob(act).sum(axis=-1)
        logp_pi -= (2 * (np.log(2) - act - torch.nn.functional.softplus(-2 * act))).sum(
            axis=1
        )
        return logp_pi

    def forward(self, obs, act=None, with_logprob=True):
        pi_distribution = self._distribution(obs)
        pi_action = pi_distribution.rsample()
        if with_logprob:
            log_pi = self._log_prob_from_distribution(pi_distribution, pi_action)
        else:
            logp_pi = None

        pi_action_tanh = torch.tanh(pi_action)

        return pi_action_tanh, logp_pi


class Critic(ABC, nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.net = None

    @abstractmethod
    def forward(self, obs, act=None) -> torch.Tensor:
        pass


class MLPCritic(Critic):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__(obs_dim, act_dim)
        self.net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs) -> torch.Tensor:
        return torch.squeeze(self.net(obs), -1)  # Critical to ensure v has right shape.


class QCritic(Critic):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, q_num=2):
        super().__init__(obs_dim, act_dim)
        self.net_encode = mlp([obs_dim + act_dim] + list(hidden_sizes), activation)
        self.q_decode_list = []
        for q_i in range(q_num):
            q_decode = mlp([hidden_sizes[-1], 1], activation=nn.Identity)
            self.q_decode_list.append(q_decode)
            setattr(self, f"q_decode_{q_i}", q_decode)

    def forward(self, obs, act) -> torch.Tensor:
        encode_obs = self.net_encode(torch.cat([obs, act], dim=-1))
        q_values = torch.cat(
            [q_decode(encode_obs) for q_decode in self.q_decode_list], dim=-1
        )
        return q_values  # Critical to ensure q has right shape.
