from src.agent_lib.dqn import DQN
from src.agent_lib.ppo import PPO
from src.agent_lib.sac import SAC
from src.train import OffPolicyTrain, OnPolicyTrain

AGENT_MAP = {"DQN": DQN, "PPO": PPO, "SAC": SAC}
TRAINER_TYPE = {"DQN": OffPolicyTrain, "SAC": OffPolicyTrain, "PPO": OnPolicyTrain}
