from agent_lib.dqn import DQN
from agent_lib.ppo import PPO
from train.policy_train import OffPolicyTrain, OnPolicyTrain

AGENT_MAP = {"DQN": DQN, "PPO": PPO}
TRAINER_TYPE = {"DQN": OffPolicyTrain, "PPO": OnPolicyTrain}
