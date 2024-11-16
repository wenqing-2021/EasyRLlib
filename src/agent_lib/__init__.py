from src.agent_lib.dqn import DQN
from src.agent_lib.ppo import PPO
from src.agent_lib.sac import SAC
from src.agent_lib.ddpg import DDPG
from src.agent_lib.base_agent import AgentFactory

AGENT_LIB_MAP = {
    "DQN": DQN,
    "PPO": PPO,
    "SAC": SAC,
    "DDPG": DDPG,
}
