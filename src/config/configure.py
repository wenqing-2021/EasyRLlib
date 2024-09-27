from dataclasses import dataclass, field, fields
from typing import Dict, Any
from torch import nn
import yaml


@dataclass
class BaseConfig:
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        instance = cls()
        if data is None:
            return instance
        instance_fields = fields(cls)
        instance_map = {f.name: f.type for f in instance_fields}
        data_keys = data.keys()
        for key in data_keys:
            value = data[key]
            if isinstance(value, dict) and issubclass(instance_map[key], BaseConfig):
                setattr(instance, key, instance_map[key].from_dict(value))
            else:
                setattr(instance, key, value)

        return instance


@dataclass
class EnvConfig(BaseConfig):
    env_name: str = None  # the name of the environment


@dataclass
class OffPolicyTrainConfig(BaseConfig):
    update_every: int = 100  # off-policy update every n steps
    random_explor_steps: int = 1e4  # off-policy random explore steps
    soft_update_every: int = 100  # soft update target network every n steps


@dataclass
class OnPolicyTrainConfig(BaseConfig):
    update_times: int = 100
    max_ep_len: int = 1000


@dataclass
class TrainConfig(BaseConfig):
    seed: int = 1
    epochs: int = 100
    total_steps: int = 1e6
    num_envs: int = 4
    batch_size: int = 64
    buffer_size: int = 1e5
    off_policy_train_config: OffPolicyTrainConfig = None
    on_policy_train_config: OnPolicyTrainConfig = None
    render: bool = False


@dataclass
class AgentConfig(BaseConfig):
    agent_name: str = None
    hidden_sizes: list = field(default_factory=lambda: [64, 64])
    gamma: float = 0.99
    activation_name: str = "Tanh"
    activation: nn.Module = None


@dataclass
class DQNConfig(AgentConfig):
    q_lr: float = 1e-3
    epsilon: float = 0.1  # epsilon greedy exploration
    tau: float = 0.005


@dataclass
class PPOConfig(AgentConfig):
    policy_lr: float = 1e-3
    value_lr: float = 1e-3
    lam: float = 0.95
    target_kl: float = 0.01
    clip_ratio: float = 0.2


@dataclass
class RunConfig(BaseConfig):
    env_config: EnvConfig = None
    train_config: TrainConfig = None
    agent_config: AgentConfig = None
    device: str = "cpu"
    agent_config_path: str = None
    save_path: str = None
    exp_name: str = None


AGENT_MAP = {"DQN": DQNConfig, "PPO": PPOConfig}

ACTIVATION_MAP = {
    "Tanh": nn.Tanh,
    "ReLU": nn.ReLU,
    "Softmax": nn.Softmax,
    "Sigmoid": nn.Sigmoid,
}


def load_config(config_path: str = None) -> RunConfig:
    if config_path is None:
        config_path = "src/config/run_config.yaml"

    with open(config_path, "r") as f:
        config_dict: dict = yaml.safe_load(f)
        agent_yaml = open(config_dict["agent_config_path"], "r")
        agent_config_dict = yaml.safe_load(agent_yaml)
        agent_name = agent_config_dict["agent_name"]
        try:
            agent_config = AGENT_MAP[agent_name].from_dict(agent_config_dict)
        except KeyError:
            raise KeyError(
                f"Agent {agent_name} not supported. Please load agent from {list(AGENT_MAP.keys())}"
            )
        try:
            activation_func = ACTIVATION_MAP[agent_config.activation_name]
        except KeyError:
            raise KeyError(
                f"Activation function {agent_config.activation_name} not supported. Please load activation from {list(ACTIVATION_MAP.keys())}"
            )

        setattr(agent_config, "activation", activation_func)
        run_config = RunConfig.from_dict(config_dict)
        setattr(run_config, "agent_config", agent_config)

        return run_config


if __name__ == "__main__":
    run_config = load_config()
    print(run_config.agent_config.activation)
    print(run_config)
