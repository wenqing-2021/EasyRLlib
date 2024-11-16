from src.agent_lib.dqn import DQN
from src.agent_lib.ppo import PPO
from src.agent_lib.sac import SAC
from src.agent_lib.ddpg import DDPG
from src.agent_lib.base_agent import AgentFactory

AgentFactory.register_agent("DQN", DQN)
AgentFactory.register_agent("PPO", PPO)
AgentFactory.register_agent("SAC", SAC)
AgentFactory.register_agent("DDPG", DDPG)

# AGENT_MAP = {"DQN": DQN, "PPO": PPO, "SAC": SAC}
# TRAINER_TYPE = {"DQN": OffPolicyTrain, "SAC": OffPolicyTrain, "PPO": OnPolicyTrain}
