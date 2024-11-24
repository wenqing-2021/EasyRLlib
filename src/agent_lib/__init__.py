from src.agent_lib.dqn import DQN
from src.agent_lib.ppo import PPO
from src.agent_lib.sac import SAC
from src.agent_lib.ddpg import DDPG
from src.agent_lib.td3 import TD3
from src.agent_lib.a2c import A2C
from src.agent_lib.base_agent import AgentFactory

AGENT_LIB_MAP = {
    "DQN": DQN,
    "PPO": PPO,
    "SAC": SAC,
    "DDPG": DDPG,
    "TD3": TD3,
    "A2C": A2C,
}
