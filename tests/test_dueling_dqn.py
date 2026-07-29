import numpy as np
import torch
from gymnasium.spaces import Box, Discrete

from src.agent_lib.dqn import DuelingDQN
from src.common.buffer import BufferData
from src.common.networks import DuelingQNet
from src.config.configure import DQNConfig, RunConfig


class RecordingLogger:
    def __init__(self):
        self.values = {}

    def store(self, **kwargs):
        self.values.update(kwargs)


def make_config():
    agent_config = DQNConfig(
        hidden_sizes=[16, 16],
        activation=torch.nn.ReLU,
        policy_lr=1e-3,
        gamma=0.99,
        epsilon=0.0,
        tau=0.005,
        q_num=2,
    )
    return RunConfig(agent_config=agent_config, device="cpu")


def test_dueling_q_values_center_advantages():
    network = DuelingQNet(4, 3, [8, 8], torch.nn.ReLU, q_num=2)
    observations = torch.randn(5, 4)

    q_values = network(observations)
    features = network.net_encode(observations)
    state_values = torch.stack(
        [decoder(features).reshape(-1) for decoder in network.value_decode_list], dim=0
    )

    assert q_values.shape == (2, 5, 3)
    torch.testing.assert_close(q_values.mean(dim=-1), state_values)


def test_dueling_dqn_uses_dueling_network_and_learns():
    logger = RecordingLogger()
    agent = DuelingDQN(
        Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
        Discrete(2),
        make_config(),
        logger,
    )
    batch = BufferData(
        obs=torch.randn(8, 4),
        act=torch.randint(0, 2, (8,)),
        next_obs=torch.randn(8, 4),
        rew=torch.randn(8, 1),
        done=torch.zeros(8, 1),
    )
    previous_parameters = [
        parameter.detach().clone() for parameter in agent.policy.parameters()
    ]

    agent.learn(batch)

    assert isinstance(agent.policy, DuelingQNet)
    assert isinstance(agent.act(np.zeros(4, dtype=np.float32)), np.ndarray)
    assert "Loss" in logger.values
    assert any(
        not torch.equal(before, after)
        for before, after in zip(previous_parameters, agent.policy.parameters())
    )
